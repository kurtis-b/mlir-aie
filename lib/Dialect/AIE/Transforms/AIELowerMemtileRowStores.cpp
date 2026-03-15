//===- AIELowerMemtileRowStores.cpp ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include <optional>
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

struct RowBankResources {
  BufferOp buffer;
  LockPair locks;
};

struct ComputeBufferResources {
  BufferOp buffer;
  LockPair locks;
};

struct RowStoreResources {
  SmallVector<ComputeBufferResources, 2> srcBuffers;
  SmallVector<ComputeBufferResources, 2> dstBuffers;
  SmallVector<RowBankResources, 2> rowBanks;
  std::optional<BufferOp> nextIndices;
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

static SmallVector<RowBankResources, 2>
createRowBanks(OpBuilder &builder, Location loc, TileOp tile, Type rowType,
               StringRef prefix, int bufferCount) {
  SmallVector<RowBankResources, 2> rowBanks;
  rowBanks.reserve(bufferCount);
  for (int i = 0; i < bufferCount; ++i) {
    std::string bankName = prefix.str() + "_row";
    if (bufferCount != 1)
      bankName += "_" + std::to_string(i);
    rowBanks.push_back(
        {createNamedBuffer(builder, loc, rowType, tile, bankName),
         {createNamedLock(builder, loc, tile, 1, bankName + "_empty"),
          createNamedLock(builder, loc, tile, 0, bankName + "_full")}});
  }
  return rowBanks;
}

static SmallVector<ComputeBufferResources, 2>
createComputeBuffers(OpBuilder &builder, Location loc, Type elemType, TileOp tile,
                     StringRef prefix, StringRef role, int bufferCount) {
  SmallVector<ComputeBufferResources, 2> buffers;
  buffers.reserve(bufferCount);
  for (int i = 0; i < bufferCount; ++i) {
    std::string bufferName = prefix.str() + "_" + role.str();
    if (bufferCount != 1)
      bufferName += "_" + std::to_string(i);
    buffers.push_back(
        {createNamedBuffer(builder, loc, elemType, tile, bufferName),
         {createNamedLock(builder, loc, tile, 1, bufferName + "_empty"),
          createNamedLock(builder, loc, tile, 0, bufferName + "_full")}});
  }
  return buffers;
}

static int getEffectiveProduceBufferCount(MemTileRowStoreOp op) {
  return op.getComputeProduceBufferCount() > 0
             ? op.getComputeProduceBufferCount()
             : op.getComputeBufferCount();
}

static int getEffectiveConsumeBufferCount(MemTileRowStoreOp op) {
  return op.getComputeConsumeBufferCount() > 0
             ? op.getComputeConsumeBufferCount()
             : op.getComputeBufferCount();
}

static Value getBufferResult(const ComputeBufferResources &buffer) {
  return buffer.buffer->getResult(0);
}

static Value getLockResult(const LockOp &lock) {
  return lock->getResult(0);
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
  int produceBufferCount = getEffectiveProduceBufferCount(op);
  int consumeBufferCount = getEffectiveConsumeBufferCount(op);

  RowStoreResources resources{
      createComputeBuffers(builder, op.getLoc(), elemType, op.getComputeTileOp(),
                           prefix, "src", produceBufferCount),
      createComputeBuffers(builder, op.getLoc(), elemType, op.getComputeTileOp(),
                           prefix, "dst", consumeBufferCount),
      createRowBanks(builder, op.getLoc(), op.getMemTileOp(), rowType, prefix,
                     op.getBufferCount()),
      std::nullopt};
  if (produceBufferCount > 1 || consumeBufferCount > 1) {
    resources.nextIndices = createNamedBuffer(
        builder, op.getLoc(), MemRefType::get({2}, builder.getI32Type()),
        op.getComputeTileOp(), prefix + "_next_index");
  }

  FlowOp::create(builder, op.getLoc(), op.getComputeTile(), WireBundle::DMA,
                 op.getComputeMm2sChannel(), op.getMemTile(), WireBundle::DMA,
                 op.getMemtileIngressChannel());
  FlowOp::create(builder, op.getLoc(), op.getMemTile(), WireBundle::DMA,
                 op.getMemtileEgressChannel(), op.getComputeTile(),
                 WireBundle::DMA, op.getComputeS2mmChannel());

  return resources;
}

static void createComputeDMAChannel(MemOp memOp, MemTileRowStoreOp op,
                                    ArrayRef<ComputeBufferResources> buffers,
                                    DMAChannelDir dir, int channel, int len) {
  Region &body = memOp.getBody();
  Block *endBlock = findEndBlock(body);
  assert(endBlock && "expected aie.end block");
  assert(!buffers.empty() && "expected at least one compute buffer");

  OpBuilder dmaBuilder(memOp.getContext());
  Block *lastStartBlock = findLastDMAStartBlock(body, endBlock);
  Block *dmaBlock = dmaBuilder.createBlock(endBlock);
  SmallVector<Block *, 2> bdBlocks;
  bdBlocks.reserve(buffers.size());
  for (auto [index, buffer] : llvm::enumerate(buffers)) {
    (void)index;
    (void)buffer;
    bdBlocks.push_back(dmaBuilder.createBlock(endBlock));
  }

  if (!lastStartBlock && &body.front() != dmaBlock)
    dmaBlock->moveBefore(&body.front());

  dmaBuilder.setInsertionPointToStart(dmaBlock);
  DMAStartOp::create(dmaBuilder, op.getLoc(), dir, channel, /*repeat_count=*/0,
                     bdBlocks.front(), endBlock);
  if (lastStartBlock)
    lastStartBlock->getTerminator()->setSuccessor(dmaBlock, 1);

  for (auto [index, buffer] : llvm::enumerate(buffers)) {
    dmaBuilder.setInsertionPointToStart(bdBlocks[index]);
    UseLockOp::create(
        dmaBuilder, op.getLoc(),
        dir == DMAChannelDir::MM2S ? getLockResult(buffer.locks.full)
                                   : getLockResult(buffer.locks.empty),
        LockAction::AcquireGreaterEqual, 1);
    DMABDOp::create(dmaBuilder, op.getLoc(), getBufferResult(buffer), 0, len);
    UseLockOp::create(
        dmaBuilder, op.getLoc(),
        dir == DMAChannelDir::MM2S ? getLockResult(buffer.locks.empty)
                                   : getLockResult(buffer.locks.full),
        LockAction::Release, 1);
    NextBDOp::create(dmaBuilder, op.getLoc(),
                     bdBlocks[(index + 1) % buffers.size()]);
  }
}

static void createComputeDMA(DeviceOp device, MemTileRowStoreOp op,
                             RowStoreResources &resources,
                             OpBuilder &builder) {
  auto memOp = getOrCreateMemOp(device, op.getComputeTileOp(), builder);
  auto elemType = cast<MemRefType>(op.getElemType());
  int len = elemType.getNumElements();

  createComputeDMAChannel(memOp, op, resources.srcBuffers, DMAChannelDir::MM2S,
                          op.getComputeMm2sChannel(), len);
  createComputeDMAChannel(memOp, op, resources.dstBuffers, DMAChannelDir::S2MM,
                          op.getComputeS2mmChannel(), len);
}

static void createMemTileDMAChannel(MemTileDMAOp dmaOp, MemTileRowStoreOp op,
                                    SmallVectorImpl<RowBankResources> &rowBanks,
                                    DMAChannelDir dir, int channel, int partCount,
                                    int elemCount) {
  Region &body = dmaOp.getBody();
  Block *endBlock = findEndBlock(body);
  assert(endBlock && "expected aie.end block");
  assert(!rowBanks.empty() && "expected at least one row bank");

  OpBuilder dmaBuilder(dmaOp.getContext());
  Block *lastStartBlock = findLastDMAStartBlock(body, endBlock);
  Block *dmaBlock = dmaBuilder.createBlock(endBlock);
  int rowLen = partCount * elemCount;

  if (!lastStartBlock && &body.front() != dmaBlock)
    dmaBlock->moveBefore(&body.front());

  if (op.getBufferCount() == 2) {
    SmallVector<Block *, 2> bankBlocks;
    bankBlocks.reserve(rowBanks.size());
    for (auto [bankIdx, bank] : llvm::enumerate(rowBanks)) {
      (void)bankIdx;
      (void)bank;
      bankBlocks.push_back(dmaBuilder.createBlock(endBlock));
    }
    dmaBuilder.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(dmaBuilder, op.getLoc(), dir, channel,
                       /*repeat_count=*/0, bankBlocks.front(), endBlock);
    if (lastStartBlock)
      lastStartBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    for (auto [bankIdx, bank] : llvm::enumerate(rowBanks)) {
      dmaBuilder.setInsertionPointToStart(bankBlocks[bankIdx]);
      UseLockOp::create(
          dmaBuilder, op.getLoc(),
          (dir == DMAChannelDir::S2MM ? bank.locks.empty : bank.locks.full)
              .getResult(),
          LockAction::AcquireGreaterEqual, 1);
      DMABDOp::create(dmaBuilder, op.getLoc(), bank.buffer.getResult(), 0,
                      rowLen);
      UseLockOp::create(
          dmaBuilder, op.getLoc(),
          (dir == DMAChannelDir::S2MM ? bank.locks.full : bank.locks.empty)
              .getResult(),
          LockAction::Release, 1);
      NextBDOp::create(dmaBuilder, op.getLoc(),
                       bankBlocks[(bankIdx + 1) % rowBanks.size()]);
    }
    return;
  }

  SmallVector<SmallVector<Block *>, 2> bankBlocks(rowBanks.size());
  for (auto [bankIdx, bank] : llvm::enumerate(rowBanks)) {
    (void)bank;
    bankBlocks[bankIdx].reserve(partCount);
    for (int i = 0; i < partCount; ++i)
      bankBlocks[bankIdx].push_back(dmaBuilder.createBlock(endBlock));
  }
  dmaBuilder.setInsertionPointToStart(dmaBlock);
  DMAStartOp::create(dmaBuilder, op.getLoc(), dir, channel,
                     /*repeat_count=*/0, bankBlocks.front().front(), endBlock);
  if (lastStartBlock)
    lastStartBlock->getTerminator()->setSuccessor(dmaBlock, 1);

  for (auto [bankIdx, bank] : llvm::enumerate(rowBanks)) {
    for (int partIdx = 0; partIdx < partCount; ++partIdx) {
      dmaBuilder.setInsertionPointToStart(bankBlocks[bankIdx][partIdx]);
      if (partIdx == 0) {
        UseLockOp::create(
            dmaBuilder, op.getLoc(),
            (dir == DMAChannelDir::S2MM ? bank.locks.empty : bank.locks.full)
                .getResult(),
            LockAction::AcquireGreaterEqual, 1);
      }
      DMABDOp::create(dmaBuilder, op.getLoc(), bank.buffer.getResult(),
                      partIdx * elemCount, elemCount);
      if (partIdx == partCount - 1) {
        UseLockOp::create(
            dmaBuilder, op.getLoc(),
            (dir == DMAChannelDir::S2MM ? bank.locks.full : bank.locks.empty)
                .getResult(),
            LockAction::Release, 1);
      }
      Block *nextBlock =
          partIdx + 1 < partCount
              ? bankBlocks[bankIdx][partIdx + 1]
              : bankBlocks[(bankIdx + 1) % rowBanks.size()].front();
      NextBDOp::create(dmaBuilder, op.getLoc(), nextBlock);
    }
  }
}

static void createMemTileDMA(DeviceOp device, MemTileRowStoreOp op,
                             RowStoreResources &resources,
                             OpBuilder &builder) {
  auto dmaOp = getOrCreateMemTileDMAOp(device, op.getMemTileOp(), builder);
  auto elemType = cast<MemRefType>(op.getElemType());
  int partCount = op.getPartCount();
  int elemCount = elemType.getNumElements();
  createMemTileDMAChannel(dmaOp, op, resources.rowBanks, DMAChannelDir::S2MM,
                          op.getMemtileIngressChannel(), partCount, elemCount);
  createMemTileDMAChannel(dmaOp, op, resources.rowBanks, DMAChannelDir::MM2S,
                          op.getMemtileEgressChannel(), partCount, elemCount);
}

static Value getPortCounterSlot(OpBuilder &builder, Location loc,
                                ObjectFifoPort port) {
  return arith::ConstantIndexOp::create(
      builder, loc, port == ObjectFifoPort::Produce ? 0 : 1);
}

static Value loadPortCounter(OpBuilder &builder, Location loc, BufferOp nextIndices,
                             ObjectFifoPort port) {
  return memref::LoadOp::create(builder, loc, nextIndices.getResult(),
                                ValueRange{getPortCounterSlot(builder, loc, port)});
}

static void initializePortCounters(CoreOp core, BufferOp nextIndices) {
  OpBuilder builder(core.getContext());
  builder.setInsertionPointToStart(&core.getBody().front());
  Value c0 = arith::ConstantIndexOp::create(builder, core.getLoc(), 0);
  Value c1 = arith::ConstantIndexOp::create(builder, core.getLoc(), 1);
  Value zero =
      arith::ConstantOp::create(builder, core.getLoc(), builder.getI32IntegerAttr(0));
  memref::StoreOp::create(builder, core.getLoc(), zero, nextIndices.getResult(),
                          ValueRange{c0});
  memref::StoreOp::create(builder, core.getLoc(), zero, nextIndices.getResult(),
                          ValueRange{c1});
}

static void updatePortCounter(OpBuilder &builder, Location loc, BufferOp nextIndices,
                              ObjectFifoPort port, int count) {
  Value slot = getPortCounterSlot(builder, loc, port);
  Value oldCounter =
      memref::LoadOp::create(builder, loc, nextIndices.getResult(), ValueRange{slot});
  Value one = arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(1));
  Value size =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(count));
  Value sum = arith::AddIOp::create(builder, loc, oldCounter, one);
  Value isGreaterEqual = arith::CmpIOp::create(builder, loc,
                                               arith::CmpIPredicate::sge, sum,
                                               size);
  Value newCounter = arith::SelectOp::create(
      builder, loc, isGreaterEqual,
      arith::SubIOp::create(builder, loc, sum, size), sum);
  memref::StoreOp::create(builder, loc, newCounter, nextIndices.getResult(),
                          ValueRange{slot});
}

static void createAcquireSwitch(OpBuilder &builder, Location loc,
                                Value switchIndex,
                                ArrayRef<ComputeBufferResources> buffers,
                                ObjectFifoPort port, Value acquireResult) {
  SmallVector<int64_t, 4> caseValues;
  for (int i = 0; i < static_cast<int>(buffers.size()); ++i)
    caseValues.push_back(i);
  auto cases = DenseI64ArrayAttr::get(builder.getContext(), caseValues);
  auto switchOp = scf::IndexSwitchOp::create(
      builder, loc, TypeRange{acquireResult.getType()}, switchIndex, cases,
      buffers.size());

  builder.createBlock(&switchOp.getDefaultRegion());
  builder.setInsertionPointToStart(&switchOp.getDefaultBlock());
  UseLockOp::create(
      builder, loc,
      port == ObjectFifoPort::Produce
          ? getLockResult(buffers.front().locks.empty)
          : getLockResult(buffers.front().locks.full),
      LockAction::AcquireGreaterEqual, 1);
  scf::YieldOp::create(builder, loc, getBufferResult(buffers.front()));

  for (auto [index, buffer] : llvm::enumerate(buffers)) {
    builder.createBlock(&switchOp.getCaseRegions()[index]);
    builder.setInsertionPointToStart(&switchOp.getCaseBlock(index));
    UseLockOp::create(
        builder, loc,
        port == ObjectFifoPort::Produce ? getLockResult(buffer.locks.empty)
                                        : getLockResult(buffer.locks.full),
        LockAction::AcquireGreaterEqual, 1);
    scf::YieldOp::create(builder, loc, getBufferResult(buffer));
  }

  acquireResult.replaceAllUsesWith(switchOp.getResult(0));
}

static scf::IndexSwitchOp
createReleaseSwitch(OpBuilder &builder, Location loc, Value switchIndex,
                    ArrayRef<ComputeBufferResources> buffers,
                    ObjectFifoPort port) {
  SmallVector<int64_t, 4> caseValues;
  for (int i = 0; i < static_cast<int>(buffers.size()); ++i)
    caseValues.push_back(i);
  auto cases = DenseI64ArrayAttr::get(builder.getContext(), caseValues);
  auto switchOp = scf::IndexSwitchOp::create(builder, loc, TypeRange{}, switchIndex,
                                             cases, buffers.size());

  builder.createBlock(&switchOp.getDefaultRegion());
  builder.setInsertionPointToStart(&switchOp.getDefaultBlock());
  UseLockOp::create(
      builder, loc,
      port == ObjectFifoPort::Produce
          ? getLockResult(buffers.front().locks.full)
          : getLockResult(buffers.front().locks.empty),
      LockAction::Release, 1);
  scf::YieldOp::create(builder, loc);

  for (auto [index, buffer] : llvm::enumerate(buffers)) {
    builder.createBlock(&switchOp.getCaseRegions()[index]);
    builder.setInsertionPointToStart(&switchOp.getCaseBlock(index));
    UseLockOp::create(
        builder, loc,
        port == ObjectFifoPort::Produce ? getLockResult(buffer.locks.full)
                                        : getLockResult(buffer.locks.empty),
        LockAction::Release, 1);
    scf::YieldOp::create(builder, loc);
  }

  return switchOp;
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

  if (resources.nextIndices) {
    SmallVector<CoreOp, 1> touchedCores;
    device.walk([&](CoreOp core) {
      if (core.getTile() == rowStore.getComputeTile())
        touchedCores.push_back(core);
    });
    for (auto core : touchedCores)
      initializePortCounters(core, *resources.nextIndices);
  }

  for (auto acquire : acquires) {
    OpBuilder builder(acquire);
    auto &buffers = acquire.getPort() == ObjectFifoPort::Produce
                        ? resources.srcBuffers
                        : resources.dstBuffers;
    if (buffers.size() == 1) {
      auto &buffer = buffers.front();
      if (acquire.getPort() == ObjectFifoPort::Produce) {
        UseLockOp::create(builder, acquire.getLoc(), buffer.locks.empty.getResult(),
                          LockAction::AcquireGreaterEqual, 1);
      } else {
        UseLockOp::create(builder, acquire.getLoc(), buffer.locks.full.getResult(),
                          LockAction::AcquireGreaterEqual, 1);
      }
      acquire.getBuffer().replaceAllUsesWith(buffer.buffer.getResult());
      continue;
    }

    Value switchIndexAsInteger =
        loadPortCounter(builder, acquire.getLoc(), *resources.nextIndices,
                        acquire.getPort());
    Value switchIndex = arith::IndexCastOp::create(builder, acquire.getLoc(),
                                                   builder.getIndexType(),
                                                   switchIndexAsInteger);
    createAcquireSwitch(builder, acquire.getLoc(), switchIndex, buffers,
                        acquire.getPort(), acquire.getBuffer());
  }

  for (auto release : releases) {
    OpBuilder builder(release);
    auto &buffers = release.getPort() == ObjectFifoPort::Produce
                        ? resources.srcBuffers
                        : resources.dstBuffers;
    if (buffers.size() == 1) {
      auto &buffer = buffers.front();
      if (release.getPort() == ObjectFifoPort::Produce) {
        UseLockOp::create(builder, release.getLoc(), buffer.locks.full.getResult(),
                          LockAction::Release, 1);
      } else {
        UseLockOp::create(builder, release.getLoc(), buffer.locks.empty.getResult(),
                          LockAction::Release, 1);
      }
      continue;
    }

    Value switchIndexAsInteger =
        loadPortCounter(builder, release.getLoc(), *resources.nextIndices,
                        release.getPort());
    Value switchIndex = arith::IndexCastOp::create(builder, release.getLoc(),
                                                   builder.getIndexType(),
                                                   switchIndexAsInteger);
    auto switchOp = createReleaseSwitch(builder, release.getLoc(), switchIndex,
                                        buffers, release.getPort());
    builder.setInsertionPointAfter(switchOp);
    updatePortCounter(builder, release.getLoc(), *resources.nextIndices,
                      release.getPort(), buffers.size());
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
    registry.insert<AIEDialect, arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
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
