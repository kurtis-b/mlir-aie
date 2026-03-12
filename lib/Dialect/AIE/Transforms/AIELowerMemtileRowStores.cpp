//===- AIELowerMemtileRowStores.cpp ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

#include <set>
#include <tuple>

#include "llvm/ADT/STLExtras.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIELOWERMEMTILEROWSTORES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-lower-memtile-row-stores"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct LockPair {
  LockOp empty;
  LockOp full;
};

struct RowStoreResources {
  BufferOp srcBuffer;
  BufferOp dstBuffer;
  BufferOp rowBuffer;
  LockPair srcLocks;
  LockPair dstLocks;
  LockPair rowLocks;
};

using ChannelKey = std::tuple<int, int, int, int>;

static Block *findEndBlock(Region &region) {
  for (Block &block : region)
    if (!block.getOps<EndOp>().empty())
      return &block;
  return nullptr;
}

static Operation *findTopLevelInsertionAnchor(DeviceOp device) {
  for (Operation &op : device.getBody()->without_terminator())
    if (isa<CoreOp, MemOp, MemTileDMAOp, ShimDMAOp>(op))
      return &op;
  return device.getBody()->getTerminator();
}

static MemOp getOrCreateMemOp(DeviceOp device, TileOp tile, OpBuilder &builder) {
  for (auto memOp : device.getOps<MemOp>())
    if (memOp.getTile() == tile.getResult())
      return memOp;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(device.getBody()->getTerminator());
  auto memOp = MemOp::create(builder, builder.getUnknownLoc(), tile);
  builder.setInsertionPointToStart(&memOp.getBody().emplaceBlock());
  EndOp::create(builder, builder.getUnknownLoc());
  return memOp;
}

static MemTileDMAOp getOrCreateMemTileDMAOp(DeviceOp device, TileOp tile,
                                            OpBuilder &builder) {
  for (auto dmaOp : device.getOps<MemTileDMAOp>())
    if (dmaOp.getTile() == tile.getResult())
      return dmaOp;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(device.getBody()->getTerminator());
  auto dmaOp = MemTileDMAOp::create(builder, builder.getUnknownLoc(), tile);
  builder.setInsertionPointToStart(&dmaOp.getBody().emplaceBlock());
  EndOp::create(builder, builder.getUnknownLoc());
  return dmaOp;
}

static Block *findLastDMAStartBlock(Region &body, Block *endBlock) {
  Block *lastStartBlock = nullptr;
  for (Block &block : body) {
    auto start = dyn_cast<DMAStartOp>(block.getTerminator());
    if (start && start.getChain() == endBlock)
      lastStartBlock = &block;
  }
  return lastStartBlock;
}

static BufferOp createNamedBuffer(OpBuilder &builder, Location loc, Type type,
                                  TileOp tile, StringRef name) {
  return BufferOp::create(builder, loc, type, tile, builder.getStringAttr(name),
                          /*address=*/nullptr, /*initial_value=*/nullptr,
                          /*mem_bank=*/nullptr);
}

static LockOp createNamedLock(OpBuilder &builder, Location loc, TileOp tile,
                              int init, StringRef name) {
  return LockOp::create(builder, loc, tile, IntegerAttr(),
                        builder.getI32IntegerAttr(init),
                        builder.getStringAttr(name));
}

static RowStoreResources createResources(DeviceOp device, MemTileRowStoreOp op,
                                         OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(findTopLevelInsertionAnchor(device));

  auto elemType = cast<MemRefType>(op.getElemType());
  int64_t elemCount = elemType.getNumElements();
  int64_t rowCount = static_cast<int64_t>(op.getPartCount()) * elemCount;
  auto rowType = MemRefType::get({rowCount}, elemType.getElementType());
  std::string prefix = op.name().str();

  RowStoreResources resources{
      createNamedBuffer(builder, op.getLoc(), elemType, op.getComputeTileOp(),
                        prefix + "_src"),
      createNamedBuffer(builder, op.getLoc(), elemType, op.getComputeTileOp(),
                        prefix + "_dst"),
      createNamedBuffer(builder, op.getLoc(), rowType, op.getMemTileOp(),
                        prefix + "_row"),
      {createNamedLock(builder, op.getLoc(), op.getComputeTileOp(), 1,
                       prefix + "_src_empty"),
       createNamedLock(builder, op.getLoc(), op.getComputeTileOp(), 0,
                       prefix + "_src_full")},
      {createNamedLock(builder, op.getLoc(), op.getComputeTileOp(), 1,
                       prefix + "_dst_empty"),
       createNamedLock(builder, op.getLoc(), op.getComputeTileOp(), 0,
                       prefix + "_dst_full")},
      {createNamedLock(builder, op.getLoc(), op.getMemTileOp(), 1,
                       prefix + "_row_empty"),
       createNamedLock(builder, op.getLoc(), op.getMemTileOp(), 0,
                       prefix + "_row_full")}};

  FlowOp::create(builder, op.getLoc(), op.getComputeTile(), WireBundle::DMA,
                 op.getComputeMm2sChannel(), op.getMemTile(), WireBundle::DMA,
                 op.getMemtileIngressChannel());
  FlowOp::create(builder, op.getLoc(), op.getMemTile(), WireBundle::DMA,
                 op.getMemtileEgressChannel(), op.getComputeTile(),
                 WireBundle::DMA, op.getComputeS2mmChannel());

  return resources;
}

static void createComputeDMA(DeviceOp device, MemTileRowStoreOp op,
                             RowStoreResources &resources,
                             OpBuilder &builder) {
  auto memOp = getOrCreateMemOp(device, op.getComputeTileOp(), builder);
  auto elemType = cast<MemRefType>(op.getElemType());
  int len = elemType.getNumElements();

  {
    Region &body = memOp.getBody();
    Block *endBlock = findEndBlock(body);
    assert(endBlock && "expected aie.end block");

    OpBuilder dmaBuilder(memOp.getContext());
    Block *lastStartBlock = findLastDMAStartBlock(body, endBlock);
    Block *dmaBlock = dmaBuilder.createBlock(endBlock);
    Block *bdBlock = dmaBuilder.createBlock(endBlock);
    if (!lastStartBlock && &body.front() != dmaBlock)
      dmaBlock->moveBefore(&body.front());
    dmaBuilder.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(dmaBuilder, op.getLoc(), DMAChannelDir::MM2S,
                       op.getComputeMm2sChannel(), /*repeat_count=*/0, bdBlock,
                       endBlock);
    if (lastStartBlock)
      lastStartBlock->getTerminator()->setSuccessor(dmaBlock, 1);
    dmaBuilder.setInsertionPointToStart(bdBlock);
    UseLockOp::create(dmaBuilder, op.getLoc(), resources.srcLocks.full.getResult(),
                      LockAction::AcquireGreaterEqual, 1);
    DMABDOp::create(dmaBuilder, op.getLoc(), resources.srcBuffer.getResult(), 0,
                    len);
    UseLockOp::create(dmaBuilder, op.getLoc(),
                      resources.srcLocks.empty.getResult(),
                      LockAction::Release, 1);
    NextBDOp::create(dmaBuilder, op.getLoc(), bdBlock);
  }

  {
    Region &body = memOp.getBody();
    Block *endBlock = findEndBlock(body);
    assert(endBlock && "expected aie.end block");

    OpBuilder dmaBuilder(memOp.getContext());
    Block *lastStartBlock = findLastDMAStartBlock(body, endBlock);
    Block *dmaBlock = dmaBuilder.createBlock(endBlock);
    Block *bdBlock = dmaBuilder.createBlock(endBlock);
    if (!lastStartBlock && &body.front() != dmaBlock)
      dmaBlock->moveBefore(&body.front());
    dmaBuilder.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(dmaBuilder, op.getLoc(), DMAChannelDir::S2MM,
                       op.getComputeS2mmChannel(), /*repeat_count=*/0, bdBlock,
                       endBlock);
    if (lastStartBlock)
      lastStartBlock->getTerminator()->setSuccessor(dmaBlock, 1);
    dmaBuilder.setInsertionPointToStart(bdBlock);
    UseLockOp::create(dmaBuilder, op.getLoc(),
                      resources.dstLocks.empty.getResult(),
                      LockAction::AcquireGreaterEqual, 1);
    DMABDOp::create(dmaBuilder, op.getLoc(), resources.dstBuffer.getResult(), 0,
                    len);
    UseLockOp::create(dmaBuilder, op.getLoc(), resources.dstLocks.full.getResult(),
                      LockAction::Release, 1);
    NextBDOp::create(dmaBuilder, op.getLoc(), bdBlock);
  }
}

static void createMemTileDMA(DeviceOp device, MemTileRowStoreOp op,
                             RowStoreResources &resources,
                             OpBuilder &builder) {
  auto dmaOp = getOrCreateMemTileDMAOp(device, op.getMemTileOp(), builder);
  auto elemType = cast<MemRefType>(op.getElemType());
  int partCount = op.getPartCount();
  int elemCount = elemType.getNumElements();

  {
    Region &body = dmaOp.getBody();
    Block *endBlock = findEndBlock(body);
    assert(endBlock && "expected aie.end block");

    OpBuilder dmaBuilder(dmaOp.getContext());
    Block *lastStartBlock = findLastDMAStartBlock(body, endBlock);
    Block *dmaBlock = dmaBuilder.createBlock(endBlock);
    SmallVector<Block *> bdBlocks;
    bdBlocks.reserve(partCount);
    for (int i = 0; i < partCount; ++i)
      bdBlocks.push_back(dmaBuilder.createBlock(endBlock));
    if (!lastStartBlock && &body.front() != dmaBlock)
      dmaBlock->moveBefore(&body.front());
    dmaBuilder.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(dmaBuilder, op.getLoc(), DMAChannelDir::S2MM,
                       op.getMemtileIngressChannel(), /*repeat_count=*/0,
                       bdBlocks.front(), endBlock);
    if (lastStartBlock)
      lastStartBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    for (int i = 0; i < partCount; ++i) {
      dmaBuilder.setInsertionPointToStart(bdBlocks[i]);
      if (i == 0) {
        UseLockOp::create(dmaBuilder, op.getLoc(),
                          resources.rowLocks.empty.getResult(),
                          LockAction::AcquireGreaterEqual, 1);
      }
      DMABDOp::create(dmaBuilder, op.getLoc(), resources.rowBuffer.getResult(),
                      i * elemCount, elemCount);
      if (i == partCount - 1) {
        UseLockOp::create(dmaBuilder, op.getLoc(),
                          resources.rowLocks.full.getResult(),
                          LockAction::Release, 1);
      }
      NextBDOp::create(dmaBuilder, op.getLoc(),
                       i + 1 < partCount ? bdBlocks[i + 1] : bdBlocks.front());
    }
  }

  {
    Region &body = dmaOp.getBody();
    Block *endBlock = findEndBlock(body);
    assert(endBlock && "expected aie.end block");

    OpBuilder dmaBuilder(dmaOp.getContext());
    Block *lastStartBlock = findLastDMAStartBlock(body, endBlock);
    Block *dmaBlock = dmaBuilder.createBlock(endBlock);
    SmallVector<Block *> bdBlocks;
    bdBlocks.reserve(partCount);
    for (int i = 0; i < partCount; ++i)
      bdBlocks.push_back(dmaBuilder.createBlock(endBlock));
    if (!lastStartBlock && &body.front() != dmaBlock)
      dmaBlock->moveBefore(&body.front());
    dmaBuilder.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(dmaBuilder, op.getLoc(), DMAChannelDir::MM2S,
                       op.getMemtileEgressChannel(), /*repeat_count=*/0,
                       bdBlocks.front(), endBlock);
    if (lastStartBlock)
      lastStartBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    for (int i = 0; i < partCount; ++i) {
      dmaBuilder.setInsertionPointToStart(bdBlocks[i]);
      if (i == 0) {
        UseLockOp::create(dmaBuilder, op.getLoc(),
                          resources.rowLocks.full.getResult(),
                          LockAction::AcquireGreaterEqual, 1);
      }
      DMABDOp::create(dmaBuilder, op.getLoc(), resources.rowBuffer.getResult(),
                      i * elemCount, elemCount);
      if (i == partCount - 1) {
        UseLockOp::create(dmaBuilder, op.getLoc(),
                          resources.rowLocks.empty.getResult(),
                          LockAction::Release, 1);
      }
      NextBDOp::create(dmaBuilder, op.getLoc(),
                       i + 1 < partCount ? bdBlocks[i + 1] : bdBlocks.front());
    }
  }
}

static void rewriteCoreAccesses(DeviceOp device, MemTileRowStoreOp rowStore,
                                RowStoreResources &resources) {
  SmallVector<MemTileRowStoreAcquireOp> acquires;
  SmallVector<MemTileRowStoreReleaseOp> releases;

  device.walk([&](MemTileRowStoreAcquireOp acquire) {
    if (acquire.getRowStore() == rowStore)
      acquires.push_back(acquire);
  });
  device.walk([&](MemTileRowStoreReleaseOp release) {
    if (release.getRowStore() == rowStore)
      releases.push_back(release);
  });

  for (auto acquire : acquires) {
    OpBuilder builder(acquire);
    if (acquire.getPort() == ObjectFifoPort::Produce) {
      UseLockOp::create(builder, acquire.getLoc(),
                        resources.srcLocks.empty.getResult(),
                        LockAction::AcquireGreaterEqual, 1);
      acquire.getBuffer().replaceAllUsesWith(resources.srcBuffer.getResult());
    } else {
      UseLockOp::create(builder, acquire.getLoc(),
                        resources.dstLocks.full.getResult(),
                        LockAction::AcquireGreaterEqual, 1);
      acquire.getBuffer().replaceAllUsesWith(resources.dstBuffer.getResult());
    }
  }

  for (auto release : releases) {
    OpBuilder builder(release);
    if (release.getPort() == ObjectFifoPort::Produce) {
      UseLockOp::create(builder, release.getLoc(),
                        resources.srcLocks.full.getResult(),
                        LockAction::Release, 1);
    } else {
      UseLockOp::create(builder, release.getLoc(),
                        resources.dstLocks.empty.getResult(),
                        LockAction::Release, 1);
    }
  }

  for (auto acquire : llvm::reverse(acquires))
    acquire.erase();
  for (auto release : llvm::reverse(releases))
    release.erase();
}

struct AIELowerMemtileRowStoresPass
    : xilinx::AIE::impl::AIELowerMemtileRowStoresBase<
          AIELowerMemtileRowStoresPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();

    std::set<ChannelKey> usedChannels;
    auto claimChannel = [&](TileOp tile, DMAChannelDir dir, int channel,
                            Operation *owner, StringRef role) -> LogicalResult {
      ChannelKey key{tile.getCol(), tile.getRow(), static_cast<int>(dir),
                     channel};
      if (!usedChannels.insert(key).second)
        return owner->emitOpError()
               << role << " conflicts with an existing DMA start on tile ("
               << tile.getCol() << ", " << tile.getRow() << "), direction "
               << stringifyDMAChannelDir(dir) << ", channel " << channel;
      return success();
    };

    for (auto memOp : device.getOps<MemOp>()) {
      for (auto dmaStart : memOp.getBody().getOps<DMAStartOp>())
        usedChannels.insert(ChannelKey{memOp.getTileOp().getCol(),
                                       memOp.getTileOp().getRow(),
                                       static_cast<int>(dmaStart.getChannelDir()),
                                       dmaStart.getChannelIndex()});
    }
    for (auto dmaOp : device.getOps<MemTileDMAOp>()) {
      for (auto dmaStart : dmaOp.getBody().getOps<DMAStartOp>())
        usedChannels.insert(ChannelKey{dmaOp.getTileOp().getCol(),
                                       dmaOp.getTileOp().getRow(),
                                       static_cast<int>(dmaStart.getChannelDir()),
                                       dmaStart.getChannelIndex()});
    }

    auto rowStores = llvm::to_vector(device.getOps<MemTileRowStoreOp>());
    OpBuilder builder(&getContext());

    for (auto rowStore : rowStores) {
      if (failed(claimChannel(rowStore.getComputeTileOp(), DMAChannelDir::MM2S,
                              rowStore.getComputeMm2sChannel(), rowStore,
                              "compute_mm2s_channel")) ||
          failed(claimChannel(rowStore.getComputeTileOp(), DMAChannelDir::S2MM,
                              rowStore.getComputeS2mmChannel(), rowStore,
                              "compute_s2mm_channel")) ||
          failed(claimChannel(rowStore.getMemTileOp(), DMAChannelDir::S2MM,
                              rowStore.getMemtileIngressChannel(), rowStore,
                              "memtile_ingress_channel")) ||
          failed(claimChannel(rowStore.getMemTileOp(), DMAChannelDir::MM2S,
                              rowStore.getMemtileEgressChannel(), rowStore,
                              "memtile_egress_channel"))) {
        return signalPassFailure();
      }
    }

    for (auto rowStore : rowStores) {
      RowStoreResources resources = createResources(device, rowStore, builder);
      createComputeDMA(device, rowStore, resources, builder);
      createMemTileDMA(device, rowStore, resources, builder);
      rewriteCoreAccesses(device, rowStore, resources);
      rowStore.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIELowerMemtileRowStoresPass() {
  return std::make_unique<AIELowerMemtileRowStoresPass>();
}
