//===- AIEDesignToJSON.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <set>
#include <algorithm>
#include <string>
#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {

namespace {

using Coord = std::pair<int, int>;

struct TileSummary {
  bool declared = false;
  bool corePresent = false;
  bool memPresent = false;
  bool switchboxPresent = false;
  bool shimMuxPresent = false;
  bool memTileDmaPresent = false;
  bool shimDmaPresent = false;
  std::vector<std::string> coreIDs;
  std::vector<std::string> buffers;
  std::vector<std::string> locks;
};

struct GroupSummary {
  std::string kind;
  std::string name;
  std::set<Coord> tiles;
  std::set<std::string> buffers;
  std::set<std::string> locks;
  std::set<std::string> dmas;
  std::set<std::string> streams;
  std::set<std::string> packetFlows;
};

struct BlockSummary {
  llvm::json::Object json;
  std::set<std::string> bufferNames;
  std::set<std::string> lockNames;
  std::optional<std::string> nextBlock;
};

struct DMAChannelSummary {
  std::string direction;
  int channelIndex = 0;
  std::set<std::string> bufferNames;
  std::set<std::string> lockNames;
};

struct DMAContainerSummary {
  llvm::json::Object json;
  Coord coord;
  std::string id;
  std::set<std::string> bufferNames;
  std::set<std::string> lockNames;
  std::set<std::pair<std::string, int>> channels;
  std::vector<DMAChannelSummary> channelSummaries;
};

using SourceKey = std::tuple<int, int, int, int>;

static Coord getCoord(TileOp tileOp) {
  return {tileOp.colIndex(), tileOp.rowIndex()};
}

static std::string getTileKind(const AIETargetModel &model, int col, int row) {
  if (model.isCoreTile(col, row))
    return "core";
  if (model.isMemTile(col, row))
    return "mem";
  if (model.isShimNOCTile(col, row))
    return "shim_noc";
  if (model.isShimPLTile(col, row))
    return "shim_pl";
  return "unknown";
}

static llvm::json::Array toStringArray(const std::vector<std::string> &values) {
  llvm::json::Array array;
  for (const std::string &value : values)
    array.push_back(value);
  return array;
}

static std::string stringifyType(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << type;
  return text;
}

static std::string stringifyAttribute(Attribute attr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << attr;
  return text;
}

static std::string stringifyRegion(Region &region) {
  std::string text;
  llvm::raw_string_ostream os(text);
  bool first = true;
  for (Block &block : region) {
    if (!first)
      os << "\n";
    first = false;
    block.print(os);
  }
  return text;
}

static std::string getBufferName(Value buffer) {
  if (Operation *def = buffer.getDefiningOp()) {
    if (auto bufferOp = dyn_cast<BufferOp>(def)) {
      if (bufferOp.hasName())
        return bufferOp.name().str();
      TileOp tileOp = bufferOp.getTileOp();
      return llvm::formatv("buffer_{0}_{1}", tileOp.colIndex(), tileOp.rowIndex())
          .str();
    }
    if (auto externalBufferOp = dyn_cast<ExternalBufferOp>(def)) {
      if (externalBufferOp.hasName())
        return externalBufferOp.name().str();
      return "external_buffer";
    }
  }
  return "value";
}

static bool isAllDigits(llvm::StringRef value) {
  return !value.empty() &&
         llvm::all_of(value, [](char c) { return c >= '0' && c <= '9'; });
}

static llvm::StringRef dropTrailingNumericSuffix(llvm::StringRef value) {
  size_t pos = value.rfind('_');
  if (pos == llvm::StringRef::npos)
    return value;
  llvm::StringRef suffix = value.substr(pos + 1);
  if (!isAllDigits(suffix))
    return value;
  return value.substr(0, pos);
}

static std::optional<std::pair<std::string, std::string>>
classifyRowStoreName(llvm::StringRef name) {
  if (name.ends_with("_next_index"))
    return std::pair<std::string, std::string>(
        "row_store", name.drop_back(11).str());

  llvm::StringRef stem = name;
  if (stem.ends_with("_empty"))
    stem = stem.drop_back(6);
  else if (stem.ends_with("_full"))
    stem = stem.drop_back(5);

  stem = dropTrailingNumericSuffix(stem);
  for (llvm::StringRef role : {"_src", "_dst", "_row"}) {
    if (stem.ends_with(role))
      return std::pair<std::string, std::string>(
          "row_store", stem.drop_back(role.size()).str());
  }
  return std::nullopt;
}

static std::optional<std::pair<std::string, std::string>>
classifyObjectFifoName(llvm::StringRef name) {
  for (llvm::StringRef pattern :
       {"_cons_buff_", "_buff_", "_prod_lock_", "_cons_lock_", "_lock_"}) {
    size_t pos = name.rfind(pattern);
    if (pos == llvm::StringRef::npos)
      continue;
    llvm::StringRef suffix = name.substr(pos + pattern.size());
    if (!isAllDigits(suffix))
      continue;
    return std::pair<std::string, std::string>("objectfifo",
                                               name.substr(0, pos).str());
  }
  if (name.ends_with("_shim_alloc"))
    return std::pair<std::string, std::string>(
        "objectfifo", name.drop_back(11).str());
  return std::nullopt;
}

static std::optional<std::pair<std::string, std::string>>
classifyNamedResource(llvm::StringRef name) {
  if (auto rowStore = classifyRowStoreName(name))
    return rowStore;
  if (auto objectFifo = classifyObjectFifoName(name))
    return objectFifo;
  return std::nullopt;
}

static GroupSummary &
getOrCreateGroup(std::map<std::pair<std::string, std::string>, GroupSummary>
                     &groups,
                 llvm::StringRef kind, llvm::StringRef name) {
  auto key = std::make_pair(kind.str(), name.str());
  auto [it, inserted] = groups.try_emplace(key);
  GroupSummary &group = it->second;
  if (inserted) {
    group.kind = key.first;
    group.name = key.second;
  }
  return group;
}

static void recordResourceGroup(
    std::map<std::pair<std::string, std::string>, GroupSummary> &groups,
    llvm::StringRef resourceName, Coord coord, bool isBuffer) {
  auto groupKey = classifyNamedResource(resourceName);
  if (!groupKey)
    return;
  GroupSummary &group = getOrCreateGroup(groups, groupKey->first, groupKey->second);
  group.tiles.insert(coord);
  if (isBuffer)
    group.buffers.insert(resourceName.str());
  else
    group.locks.insert(resourceName.str());
}

static std::string getDMAID(llvm::StringRef kind, TileOp tileOp) {
  return llvm::formatv("{0}_{1}_{2}", kind, tileOp.colIndex(), tileOp.rowIndex())
      .str();
}

static std::string getStreamID(int index) {
  return llvm::formatv("route{0}", index).str();
}

static SourceKey getSourceKey(TileOp tileOp, WireBundle bundle, int channel) {
  return {tileOp.colIndex(), tileOp.rowIndex(), static_cast<int>(bundle), channel};
}

static std::string getLockName(LockOp lockOp) {
  if (lockOp.hasName())
    return lockOp.name().str();
  TileOp tileOp = lockOp.getTileOp();
  return llvm::formatv("lock_{0}_{1}", tileOp.colIndex(), tileOp.rowIndex())
      .str();
}

static llvm::json::Object buildBufferJSON(BufferOp bufferOp) {
  TileOp tileOp = bufferOp.getTileOp();
  llvm::json::Object bufferJSON;
  bufferJSON["name"] = getBufferName(bufferOp.getResult());
  bufferJSON["col"] = tileOp.colIndex();
  bufferJSON["row"] = tileOp.rowIndex();
  bufferJSON["type"] = stringifyType(bufferOp.getBuffer().getType());
  bufferJSON["allocation_bytes"] =
      static_cast<int64_t>(bufferOp.getAllocationSize());
  if (auto address = bufferOp.getAddress())
    bufferJSON["address"] = static_cast<int64_t>(*address);
  if (auto memBank = bufferOp.getMemBank())
    bufferJSON["mem_bank"] = static_cast<int64_t>(*memBank);
  return bufferJSON;
}

static llvm::json::Object buildLockJSON(LockOp lockOp) {
  TileOp tileOp = lockOp.getTileOp();
  llvm::json::Object lockJSON;
  lockJSON["name"] = getLockName(lockOp);
  lockJSON["col"] = tileOp.colIndex();
  lockJSON["row"] = tileOp.rowIndex();
  if (auto lockID = lockOp.getLockID())
    lockJSON["lock_id"] = static_cast<int64_t>(*lockID);
  if (auto init = lockOp.getInit())
    lockJSON["init"] = static_cast<int64_t>(*init);
  return lockJSON;
}

static llvm::json::Object buildPacketEndpointJSON(TileOp tileOp, WireBundle bundle,
                                                  int channel) {
  llvm::json::Object endpointJSON;
  endpointJSON["col"] = tileOp.colIndex();
  endpointJSON["row"] = tileOp.rowIndex();
  endpointJSON["bundle"] = stringifyWireBundle(bundle).str();
  endpointJSON["channel"] = static_cast<int64_t>(channel);
  return endpointJSON;
}

static std::string getGroupID(const GroupSummary &group) {
  return llvm::formatv("{0}:{1}", group.kind, group.name).str();
}

static std::string getFlowGroupName(TileOp tileOp, WireBundle bundle, int channel) {
  return llvm::formatv("flow_{0}_{1}_{2}_{3}", tileOp.colIndex(),
                       tileOp.rowIndex(), stringifyWireBundle(bundle), channel)
      .str();
}

static std::string getPacketFlowID(PacketFlowOp pktFlowOp, int index) {
  return llvm::formatv("packet_{0}_{1}", pktFlowOp.IDInt(), index).str();
}

static llvm::json::Object buildPacketFlowJSON(PacketFlowOp pktFlowOp, int index) {
  llvm::json::Object packetFlowJSON;
  packetFlowJSON["id"] = getPacketFlowID(pktFlowOp, index);
  packetFlowJSON["packet_id"] = static_cast<int64_t>(pktFlowOp.IDInt());
  if (auto keep = pktFlowOp.getKeepPktHeader())
    packetFlowJSON["keep_pkt_header"] = *keep;
  if (auto priority = pktFlowOp.getPriorityRoute())
    packetFlowJSON["priority_route"] = *priority;

  llvm::json::Array destinations;
  Region &region = pktFlowOp.getPorts();
  for (Operation &op : region.front()) {
    if (auto packetSourceOp = dyn_cast<PacketSourceOp>(op)) {
      TileOp tileOp = cast<TileOp>(packetSourceOp.getTile().getDefiningOp());
      packetFlowJSON["source"] = buildPacketEndpointJSON(
          tileOp, packetSourceOp.getBundle(), packetSourceOp.channelIndex());
      continue;
    }
    if (auto packetDestOp = dyn_cast<PacketDestOp>(op)) {
      TileOp tileOp = cast<TileOp>(packetDestOp.getTile().getDefiningOp());
      destinations.push_back(buildPacketEndpointJSON(
          tileOp, packetDestOp.getBundle(), packetDestOp.channelIndex()));
      continue;
    }
  }
  packetFlowJSON["destinations"] = std::move(destinations);
  return packetFlowJSON;
}

static std::string getCoreID(CoreOp coreOp) {
  TileOp tileOp = coreOp.getTileOp();
  return llvm::formatv("core_{0}_{1}", tileOp.colIndex(), tileOp.rowIndex())
      .str();
}

static llvm::json::Object buildCoreJSON(CoreOp coreOp) {
  TileOp tileOp = coreOp.getTileOp();
  llvm::json::Object coreJSON;
  coreJSON["id"] = getCoreID(coreOp);
  coreJSON["col"] = tileOp.colIndex();
  coreJSON["row"] = tileOp.rowIndex();
  coreJSON["stack_size"] = static_cast<int64_t>(coreOp.getStackSize());
  if (auto linkWith = coreOp.getLinkWith())
    coreJSON["link_with"] = linkWith->str();
  if (auto elfFile = coreOp.getElfFile())
    coreJSON["elf_file"] = elfFile->str();
  if (auto dynamicObjFifoLowering = coreOp.getDynamicObjfifoLowering())
    coreJSON["dynamic_objfifo_lowering"] = *dynamicObjFifoLowering;
  coreJSON["is_empty"] = coreOp.isEmpty();

  llvm::json::Array operations;
  int64_t operationCount = 0;
  for (Block &block : coreOp.getBody()) {
    for (Operation &op : block) {
      operations.push_back(op.getName().getStringRef().str());
      ++operationCount;
    }
  }
  coreJSON["operation_count"] = operationCount;
  coreJSON["operations"] = std::move(operations);
  coreJSON["body"] = stringifyRegion(coreOp.getBody());
  return coreJSON;
}

static llvm::DenseMap<Block *, std::string> labelBlocks(Region &region) {
  llvm::DenseMap<Block *, std::string> labels;
  int index = 0;
  for (Block &block : region)
    labels[&block] = llvm::formatv("block{0}", index++).str();
  return labels;
}

static std::pair<llvm::json::Object, std::string> buildUseLockJSON(UseLockOp useLockOp) {
  LockOp lockOp = useLockOp.getLockOp();
  llvm::json::Object useLockJSON;
  useLockJSON["lock"] = getLockName(lockOp);
  useLockJSON["action"] = stringifyLockAction(useLockOp.getAction()).str();
  useLockJSON["value"] = static_cast<int64_t>(useLockOp.getLockValue());
  useLockJSON["blocking"] = static_cast<int64_t>(useLockOp.getTimeout());
  if (auto lockID = lockOp.getLockID())
    useLockJSON["lock_id"] = static_cast<int64_t>(*lockID);
  return {std::move(useLockJSON), getLockName(lockOp)};
}

static std::pair<llvm::json::Object, std::string> buildDMABDJSON(DMABDOp bdOp) {
  std::string bufferName = getBufferName(bdOp.getBuffer());
  llvm::json::Object bdJSON;
  bdJSON["buffer"] = bufferName;
  bdJSON["buffer_type"] = stringifyType(bdOp.getBuffer().getType());
  bdJSON["offset"] = static_cast<int64_t>(bdOp.getOffset());
  bdJSON["offset_bytes"] = static_cast<int64_t>(bdOp.getOffsetInBytes());
  bdJSON["length_bytes"] = static_cast<int64_t>(bdOp.getLenInBytes());
  if (auto len = bdOp.getLen())
    bdJSON["length"] = static_cast<int64_t>(*len);
  if (Attribute dimensions = bdOp->getAttr("dimensions"))
    bdJSON["dimensions"] = stringifyAttribute(dimensions);
  if (Attribute padDimensions = bdOp->getAttr("pad_dimensions"))
    bdJSON["pad_dimensions"] = stringifyAttribute(padDimensions);
  if (bdOp.getPadValue() != 0)
    bdJSON["pad_value"] = static_cast<int64_t>(bdOp.getPadValue());
  if (auto bdID = bdOp.getBdId())
    bdJSON["bd_id"] = static_cast<int64_t>(*bdID);
  if (auto nextBDID = bdOp.getNextBdId())
    bdJSON["next_bd_id"] = static_cast<int64_t>(*nextBDID);
  if (bdOp.getBurstLength() != 0)
    bdJSON["burst_length"] = static_cast<int64_t>(bdOp.getBurstLength());
  if (auto packet = bdOp.getPacket())
    bdJSON["packet"] = stringifyAttribute(*packet);
  return {std::move(bdJSON), bufferName};
}

static BlockSummary buildBlockJSON(
    Block &block, const llvm::DenseMap<Block *, std::string> &blockLabels) {
  BlockSummary summary;
  llvm::json::Object &blockJSON = summary.json;
  blockJSON["id"] = blockLabels.lookup(&block);

  llvm::json::Array locks;
  llvm::json::Object dmaBD;

  for (Operation &op : block) {
    if (auto useLockOp = dyn_cast<UseLockOp>(op)) {
      auto [useLockJSON, lockName] = buildUseLockJSON(useLockOp);
      locks.push_back(std::move(useLockJSON));
      summary.lockNames.insert(std::move(lockName));
      continue;
    }
    if (auto bdOp = dyn_cast<DMABDOp>(op)) {
      auto [bdJSON, bufferName] = buildDMABDJSON(bdOp);
      dmaBD = std::move(bdJSON);
      summary.bufferNames.insert(std::move(bufferName));
      continue;
    }
    if (auto nextBDOp = dyn_cast<NextBDOp>(op)) {
      blockJSON["terminator"] = "next_bd";
      blockJSON["next"] = blockLabels.lookup(nextBDOp.getDest());
      summary.nextBlock = blockLabels.lookup(nextBDOp.getDest());
      continue;
    }
    if (isa<EndOp>(op)) {
      blockJSON["terminator"] = "end";
      continue;
    }
  }

  if (!locks.empty())
    blockJSON["locks"] = std::move(locks);
  if (!dmaBD.empty())
    blockJSON["dma_bd"] = std::move(dmaBD);
  return summary;
}

static BlockSummary buildDMABlockJSON(Block &block, llvm::StringRef blockID) {
  BlockSummary summary;
  llvm::json::Object &blockJSON = summary.json;
  blockJSON["id"] = blockID.str();

  llvm::json::Array locks;
  llvm::json::Object dmaBD;

  for (Operation &op : block) {
    if (auto useLockOp = dyn_cast<UseLockOp>(op)) {
      auto [useLockJSON, lockName] = buildUseLockJSON(useLockOp);
      locks.push_back(std::move(useLockJSON));
      summary.lockNames.insert(std::move(lockName));
      continue;
    }
    if (auto bdOp = dyn_cast<DMABDOp>(op)) {
      auto [bdJSON, bufferName] = buildDMABDJSON(bdOp);
      dmaBD = std::move(bdJSON);
      summary.bufferNames.insert(std::move(bufferName));
      continue;
    }
  }

  if (!locks.empty())
    blockJSON["locks"] = std::move(locks);
  if (!dmaBD.empty())
    blockJSON["dma_bd"] = std::move(dmaBD);
  return summary;
}

template <typename DMAOpTy>
static DMAContainerSummary buildDMAContainerJSON(DMAOpTy dmaOp,
                                                 llvm::StringRef kind) {
  TileOp tileOp = dmaOp.getTileOp();
  DMAContainerSummary summary;
  llvm::json::Object &dmaJSON = summary.json;
  summary.coord = getCoord(tileOp);
  summary.id = getDMAID(kind, tileOp);
  dmaJSON["id"] = summary.id;
  dmaJSON["kind"] = kind.str();
  dmaJSON["col"] = tileOp.colIndex();
  dmaJSON["row"] = tileOp.rowIndex();

  Region &region = dmaOp.getBody();
  llvm::DenseMap<Block *, std::string> blockLabels = labelBlocks(region);
  bool hasDMAStarts = !region.getOps<DMAStartOp>().empty();

  std::map<std::string, BlockSummary> blockSummariesByID;
  llvm::json::Array blocks;
  for (Block &block : region) {
    bool shouldSummarize =
        hasDMAStarts
            ? llvm::any_of(block, [](Operation &op) {
                return isa<UseLockOp, DMABDOp, NextBDOp, EndOp>(op);
              })
            : llvm::any_of(block, [](Operation &op) {
                return isa<UseLockOp, DMABDOp, NextBDOp>(op);
              });
    if (shouldSummarize) {
      std::string blockID = blockLabels.lookup(&block);
      BlockSummary blockSummary = buildBlockJSON(block, blockLabels);
      blockSummariesByID[blockID] = blockSummary;
      summary.bufferNames.insert(blockSummary.bufferNames.begin(),
                                 blockSummary.bufferNames.end());
      summary.lockNames.insert(blockSummary.lockNames.begin(),
                               blockSummary.lockNames.end());
      blocks.push_back(std::move(blockSummary.json));
    }
  }
  llvm::json::Array channels;
  for (Operation &op : region.getOps()) {
    if (auto dmaStartOp = dyn_cast<DMAStartOp>(op)) {
      DMAChannelSummary channelSummary;
      channelSummary.direction =
          stringifyDMAChannelDir(dmaStartOp.getChannelDir()).str();
      channelSummary.channelIndex = dmaStartOp.getChannelIndex();

      llvm::SmallVector<std::string, 8> worklist;
      std::set<std::string> visited;
      worklist.push_back(blockLabels.lookup(dmaStartOp.getDest()));
      while (!worklist.empty()) {
        std::string blockID = worklist.pop_back_val();
        if (blockID.empty() || !visited.insert(blockID).second)
          continue;
        auto blockIt = blockSummariesByID.find(blockID);
        if (blockIt == blockSummariesByID.end())
          continue;
        const BlockSummary &blockSummary = blockIt->second;
        channelSummary.bufferNames.insert(blockSummary.bufferNames.begin(),
                                          blockSummary.bufferNames.end());
        channelSummary.lockNames.insert(blockSummary.lockNames.begin(),
                                        blockSummary.lockNames.end());
        if (blockSummary.nextBlock)
          worklist.push_back(*blockSummary.nextBlock);
      }

      llvm::json::Object channelJSON;
      channelJSON["style"] = "dma_start";
      channelJSON["direction"] = channelSummary.direction;
      channelJSON["channel_index"] =
          static_cast<int64_t>(dmaStartOp.getChannelIndex());
      channelJSON["repeat_count"] =
          static_cast<int64_t>(dmaStartOp.getRepeatCount());
      channelJSON["dest"] = blockLabels.lookup(dmaStartOp.getDest());
      channelJSON["chain"] = blockLabels.lookup(dmaStartOp.getChain());
      channelJSON["buffers"] = toStringArray(std::vector<std::string>(
          channelSummary.bufferNames.begin(), channelSummary.bufferNames.end()));
      channelJSON["locks"] = toStringArray(std::vector<std::string>(
          channelSummary.lockNames.begin(), channelSummary.lockNames.end()));
      summary.channels.insert({channelSummary.direction, dmaStartOp.getChannelIndex()});
      summary.channelSummaries.push_back(std::move(channelSummary));
      channels.push_back(std::move(channelJSON));
    }
    if (auto structuredDMAOp = dyn_cast<DMAOp>(op)) {
      DMAChannelSummary channelSummary;
      channelSummary.direction =
          stringifyDMAChannelDir(structuredDMAOp.getChannelDir()).str();
      channelSummary.channelIndex = structuredDMAOp.getChannelIndex();

      llvm::json::Array blockIDs;
      int regionIndex = 0;
      for (Region &bdRegion : structuredDMAOp.getBds()) {
        Block &bdBlock = bdRegion.front();
        std::string blockID =
            llvm::formatv("dma_{0}_{1}_bd{2}", channelSummary.direction,
                          structuredDMAOp.getChannelIndex(), regionIndex++)
                .str();
        BlockSummary blockSummary = buildDMABlockJSON(bdBlock, blockID);
        summary.bufferNames.insert(blockSummary.bufferNames.begin(),
                                   blockSummary.bufferNames.end());
        summary.lockNames.insert(blockSummary.lockNames.begin(),
                                 blockSummary.lockNames.end());
        channelSummary.bufferNames.insert(blockSummary.bufferNames.begin(),
                                          blockSummary.bufferNames.end());
        channelSummary.lockNames.insert(blockSummary.lockNames.begin(),
                                        blockSummary.lockNames.end());
        blockSummariesByID[blockID] = blockSummary;
        blockIDs.push_back(blockID);
        blocks.push_back(std::move(blockSummary.json));
      }

      llvm::json::Object channelJSON;
      channelJSON["style"] = "dma";
      channelJSON["direction"] = channelSummary.direction;
      channelJSON["channel_index"] =
          static_cast<int64_t>(structuredDMAOp.getChannelIndex());
      channelJSON["loop"] = structuredDMAOp.getLoop();
      channelJSON["repeat_count"] =
          static_cast<int64_t>(structuredDMAOp.getRepeatCount());
      if (auto symName =
              structuredDMAOp->getAttrOfType<StringAttr>(
                  SymbolTable::getSymbolAttrName()))
        channelJSON["symbol_name"] = symName.str();
      channelJSON["block_ids"] = std::move(blockIDs);
      channelJSON["buffers"] = toStringArray(std::vector<std::string>(
          channelSummary.bufferNames.begin(), channelSummary.bufferNames.end()));
      channelJSON["locks"] = toStringArray(std::vector<std::string>(
          channelSummary.lockNames.begin(), channelSummary.lockNames.end()));
      summary.channels.insert(
          {channelSummary.direction, structuredDMAOp.getChannelIndex()});
      summary.channelSummaries.push_back(std::move(channelSummary));
      channels.push_back(std::move(channelJSON));
    }
  }
  dmaJSON["bd_blocks"] = std::move(blocks);
  dmaJSON["channels"] = std::move(channels);
  return summary;
}

static llvm::json::Array coordArray(const std::set<Coord> &coords) {
  llvm::json::Array array;
  for (const Coord &coord : coords) {
    llvm::json::Object tileJSON;
    tileJSON["col"] = coord.first;
    tileJSON["row"] = coord.second;
    array.push_back(std::move(tileJSON));
  }
  return array;
}

static llvm::json::Array buildGroupJSON(
    const std::map<std::pair<std::string, std::string>, GroupSummary> &groups) {
  llvm::json::Array array;
  for (const auto &entry : groups) {
    const GroupSummary &group = entry.second;
    llvm::json::Object groupJSON;
    groupJSON["id"] =
        llvm::formatv("{0}:{1}", group.kind, group.name).str();
    groupJSON["kind"] = group.kind;
    groupJSON["name"] = group.name;
    groupJSON["tiles"] = coordArray(group.tiles);
    groupJSON["buffers"] = toStringArray(
        std::vector<std::string>(group.buffers.begin(), group.buffers.end()));
    groupJSON["locks"] = toStringArray(
        std::vector<std::string>(group.locks.begin(), group.locks.end()));
    groupJSON["dmas"] = toStringArray(
        std::vector<std::string>(group.dmas.begin(), group.dmas.end()));
    groupJSON["streams"] = toStringArray(
        std::vector<std::string>(group.streams.begin(), group.streams.end()));
    groupJSON["packet_flows"] = toStringArray(std::vector<std::string>(
        group.packetFlows.begin(), group.packetFlows.end()));
    array.push_back(std::move(groupJSON));
  }
  return array;
}

static void recordDMAGroupMembership(
    std::map<std::pair<std::string, std::string>, GroupSummary> &groups,
    const DMAContainerSummary &dmaSummary) {
  for (auto &[key, group] : groups) {
    (void)key;
    bool matches = llvm::any_of(group.buffers, [&](const std::string &buffer) {
      return dmaSummary.bufferNames.find(buffer) != dmaSummary.bufferNames.end();
    }) || llvm::any_of(group.locks, [&](const std::string &lock) {
      return dmaSummary.lockNames.find(lock) != dmaSummary.lockNames.end();
    });
    if (matches)
      group.dmas.insert(dmaSummary.id);
  }
}

struct GroupMatchSummary {
  std::string id;
  int score = 0;
  bool sourceDMAMatch = false;
  int matchedDestinations = 0;
  int totalDestinations = 0;
};

struct CircuitProvenanceSummary {
  std::string status;
  std::string method;
  int selectedGroupCount = 0;
  int consideredGroupCount = 0;
  std::optional<int> bestScore;
  bool usedFallbackFlowGroup = false;
};

static bool groupHasDMAChannel(
    const GroupSummary &group,
    const std::map<std::string, DMAContainerSummary> &dmaSummariesByID,
    Coord coord, llvm::StringRef direction, int channel) {
  for (const std::string &dmaID : group.dmas) {
    auto it = dmaSummariesByID.find(dmaID);
    if (it == dmaSummariesByID.end())
      continue;
    const DMAContainerSummary &dmaSummary = it->second;
    if (dmaSummary.coord != coord)
      continue;
    for (const DMAChannelSummary &channelSummary : dmaSummary.channelSummaries) {
      if (channelSummary.direction != direction ||
          channelSummary.channelIndex != channel)
        continue;
      bool bufferMatch = llvm::any_of(channelSummary.bufferNames,
                                      [&](const std::string &bufferName) {
                                        return group.buffers.count(bufferName);
                                      });
      bool lockMatch = llvm::any_of(channelSummary.lockNames,
                                    [&](const std::string &lockName) {
                                      return group.locks.count(lockName);
                                    });
      if (bufferMatch || lockMatch)
        return true;
    }
  }
  return false;
}

static std::vector<GroupMatchSummary> collectCircuitGroupMatches(
    const std::map<std::pair<std::string, std::string>, GroupSummary> &groups,
    const std::map<std::string, DMAContainerSummary> &dmaSummariesByID,
    TileOp source, llvm::ArrayRef<FlowOp> flowOps) {
  Coord srcCoord = getCoord(source);
  std::vector<GroupMatchSummary> matches;
  for (const auto &[key, group] : groups) {
    (void)key;
    if (group.kind != "row_store" && group.kind != "objectfifo")
      continue;
    if (group.tiles.find(srcCoord) == group.tiles.end())
      continue;
    GroupMatchSummary match;
    match.id = getGroupID(group);
    bool touchesAnyDest = false;
    for (FlowOp flowOp : flowOps) {
      TileOp dest = cast<TileOp>(flowOp.getDest().getDefiningOp());
      Coord dstCoord = getCoord(dest);
      if (group.tiles.find(dstCoord) == group.tiles.end())
        continue;
      touchesAnyDest = true;
      if (flowOp.getDestBundle() == WireBundle::DMA) {
        ++match.totalDestinations;
        if (groupHasDMAChannel(group, dmaSummariesByID, dstCoord, "S2MM",
                               flowOp.getDestChannel())) {
          ++match.matchedDestinations;
          ++match.score;
        }
      }
    }
    if (!touchesAnyDest)
      continue;
    FlowOp firstFlowOp = flowOps.front();
    if (firstFlowOp.getSourceBundle() == WireBundle::DMA &&
        groupHasDMAChannel(group, dmaSummariesByID, srcCoord, "MM2S",
                           firstFlowOp.getSourceChannel())) {
      match.sourceDMAMatch = true;
      match.score += 2;
    }
    matches.push_back(std::move(match));
  }
  llvm::sort(matches, [](const GroupMatchSummary &lhs,
                         const GroupMatchSummary &rhs) {
    if (lhs.score != rhs.score)
      return lhs.score > rhs.score;
    return lhs.id < rhs.id;
  });
  return matches;
}

static std::vector<std::string>
selectBestCircuitGroupCandidates(llvm::ArrayRef<GroupMatchSummary> matches) {
  if (matches.empty())
    return {};
  if (matches.front().score <= 0) {
    std::vector<std::string> ids;
    for (const GroupMatchSummary &match : matches)
      ids.push_back(match.id);
    return ids;
  }
  std::vector<std::string> ids;
  int bestScore = matches.front().score;
  for (const GroupMatchSummary &match : matches) {
    if (match.score != bestScore)
      break;
    ids.push_back(match.id);
  }
  return ids;
}

static llvm::json::Array buildGroupMatchesJSON(
    llvm::ArrayRef<GroupMatchSummary> matches) {
  llvm::json::Array matchArray;
  for (const GroupMatchSummary &match : matches) {
    llvm::json::Object matchJSON;
    matchJSON["id"] = match.id;
    matchJSON["score"] = static_cast<int64_t>(match.score);
    matchJSON["source_dma_match"] = match.sourceDMAMatch;
    matchJSON["matched_destinations"] =
        static_cast<int64_t>(match.matchedDestinations);
    matchJSON["total_dma_destinations"] =
        static_cast<int64_t>(match.totalDestinations);
    matchArray.push_back(std::move(matchJSON));
  }
  return matchArray;
}

static CircuitProvenanceSummary summarizeCircuitProvenance(
    llvm::ArrayRef<std::string> groupIDs,
    llvm::ArrayRef<GroupMatchSummary> matches, bool usedFallbackFlowGroup) {
  CircuitProvenanceSummary summary;
  summary.selectedGroupCount = groupIDs.size();
  summary.consideredGroupCount = matches.size();
  summary.usedFallbackFlowGroup = usedFallbackFlowGroup;

  if (usedFallbackFlowGroup) {
    summary.status = "fallback";
    summary.method = "flow_group";
    return summary;
  }

  if (!matches.empty())
    summary.bestScore = matches.front().score;

  if (!matches.empty() && matches.front().score > 0) {
    summary.status = groupIDs.size() == 1 ? "resolved" : "ambiguous";
    summary.method = "dma_channel";
    return summary;
  }

  summary.status = groupIDs.size() == 1 ? "resolved" : "ambiguous";
  summary.method = "tile_membership";
  return summary;
}

static llvm::json::Object
buildCircuitProvenanceJSON(const CircuitProvenanceSummary &summary) {
  llvm::json::Object provenanceJSON;
  provenanceJSON["status"] = summary.status;
  provenanceJSON["method"] = summary.method;
  provenanceJSON["selected_group_count"] =
      static_cast<int64_t>(summary.selectedGroupCount);
  provenanceJSON["considered_group_count"] =
      static_cast<int64_t>(summary.consideredGroupCount);
  provenanceJSON["used_fallback_flow_group"] = summary.usedFallbackFlowGroup;
  if (summary.bestScore)
    provenanceJSON["best_score"] = static_cast<int64_t>(*summary.bestScore);
  return provenanceJSON;
}

static void recordStreamGroupMembership(
    std::map<std::pair<std::string, std::string>, GroupSummary> &groups,
    llvm::ArrayRef<std::string> groupIDs, llvm::StringRef streamID) {
  for (auto &[key, group] : groups) {
    (void)key;
    if (llvm::is_contained(groupIDs, getGroupID(group)))
      group.streams.insert(streamID.str());
  }
}

static llvm::json::Array buildGroupCandidateJSON(
    llvm::ArrayRef<std::string> groupIDs) {
  llvm::json::Array groups;
  for (llvm::StringRef id : groupIDs)
    groups.push_back(id.str());
  return groups;
}

static llvm::json::Object buildCircuitStreamJSON(
    llvm::StringRef routeID, TileOp sourceTile, WireBundle sourceBundle,
    int sourceChannel, llvm::ArrayRef<FlowOp> flowOps,
    const llvm::json::Value &route, llvm::ArrayRef<std::string> groupIDs,
    llvm::ArrayRef<GroupMatchSummary> groupMatches,
    const CircuitProvenanceSummary &provenance) {
  llvm::json::Object stream;
  stream["id"] = routeID.str();
  stream["kind"] = "circuit";
  stream["source"] =
      buildPacketEndpointJSON(sourceTile, sourceBundle, sourceChannel);
  llvm::json::Array destinations;
  for (FlowOp flowOp : flowOps) {
    auto destTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    destinations.push_back(buildPacketEndpointJSON(destTile, flowOp.getDestBundle(),
                                                   flowOp.getDestChannel()));
  }
  if (flowOps.size() == 1) {
    FlowOp flowOp = flowOps.front();
    auto destTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    stream["dest"] =
        buildPacketEndpointJSON(destTile, flowOp.getDestBundle(),
                                flowOp.getDestChannel());
  }
  stream["destinations"] = std::move(destinations);
  stream["group_candidates"] = buildGroupCandidateJSON(groupIDs);
  stream["group_matches"] = buildGroupMatchesJSON(groupMatches);
  stream["provenance"] = buildCircuitProvenanceJSON(provenance);
  stream["route"] = route;
  return stream;
}

static llvm::json::Object buildPacketStreamJSON(
    llvm::StringRef routeID, PacketFlowOp pktFlowOp, int packetFlowIndex,
    const llvm::json::Value &route, llvm::StringRef groupID) {
  llvm::json::Object stream;
  stream["id"] = routeID.str();
  stream["kind"] = "packet";
  stream["packet_flow_id"] = getPacketFlowID(pktFlowOp, packetFlowIndex);

  llvm::json::Array destinations;
  Region &region = pktFlowOp.getPorts();
  for (Operation &op : region.front()) {
    if (auto packetSourceOp = dyn_cast<PacketSourceOp>(op)) {
      TileOp tileOp = cast<TileOp>(packetSourceOp.getTile().getDefiningOp());
      stream["source"] = buildPacketEndpointJSON(
          tileOp, packetSourceOp.getBundle(), packetSourceOp.channelIndex());
    } else if (auto packetDestOp = dyn_cast<PacketDestOp>(op)) {
      TileOp tileOp = cast<TileOp>(packetDestOp.getTile().getDefiningOp());
      destinations.push_back(buildPacketEndpointJSON(
          tileOp, packetDestOp.getBundle(), packetDestOp.channelIndex()));
    }
  }

  stream["destinations"] = std::move(destinations);
  stream["group_candidates"] = llvm::json::Array{groupID.str()};
  stream["route"] = route;
  return stream;
}

} // namespace

LogicalResult AIEDesignToJSON(ModuleOp module, raw_ostream &output,
                              llvm::StringRef deviceName) {
  DeviceOp targetOp = AIE::DeviceOp::getForSymbolInModule(module, deviceName);
  if (!targetOp) {
    module.emitOpError("expected AIE.device operation at toplevel");
    return failure();
  }

  const AIETargetModel &model = targetOp.getTargetModel();
  std::map<Coord, TileSummary> summaries;
  llvm::json::Array cores;
  llvm::json::Array buffers;
  llvm::json::Array locks;
  llvm::json::Array dmas;
  llvm::json::Array packetFlows;
  std::map<std::pair<std::string, std::string>, GroupSummary> groups;
  std::vector<std::string> packetFlowGroupIDs;
  std::map<std::string, DMAContainerSummary> dmaSummariesByID;

  for (TileOp tileOp : targetOp.getOps<TileOp>()) {
    Coord coord = getCoord(tileOp);
    TileSummary &summary = summaries[coord];
    summary.declared = true;
    summary.corePresent = bool(tileOp.getCoreOp());
    summary.memPresent = bool(tileOp.getMemOp());
  }
  for (CoreOp coreOp : targetOp.getOps<CoreOp>()) {
    TileOp tileOp = coreOp.getTileOp();
    summaries[getCoord(tileOp)].coreIDs.push_back(getCoreID(coreOp));
    cores.push_back(buildCoreJSON(coreOp));
  }
  for (SwitchboxOp switchboxOp : targetOp.getOps<SwitchboxOp>())
    summaries[{switchboxOp.colIndex(), switchboxOp.rowIndex()}].switchboxPresent =
        true;
  for (ShimMuxOp shimMuxOp : targetOp.getOps<ShimMuxOp>())
    summaries[{shimMuxOp.colIndex(), shimMuxOp.rowIndex()}].shimMuxPresent = true;
  for (MemTileDMAOp dmaOp : targetOp.getOps<MemTileDMAOp>())
    summaries[{dmaOp.colIndex(), dmaOp.rowIndex()}].memTileDmaPresent = true;
  for (ShimDMAOp shimDmaOp : targetOp.getOps<ShimDMAOp>())
    summaries[{shimDmaOp.colIndex(), shimDmaOp.rowIndex()}].shimDmaPresent = true;
  for (BufferOp bufferOp : targetOp.getOps<BufferOp>()) {
    TileOp tileOp = bufferOp.getTileOp();
    Coord coord = getCoord(tileOp);
    TileSummary &summary = summaries[coord];
    std::string bufferName = getBufferName(bufferOp.getResult());
    summary.buffers.push_back(bufferName);
    recordResourceGroup(groups, bufferName, coord, /*isBuffer=*/true);
    buffers.push_back(buildBufferJSON(bufferOp));
  }
  for (LockOp lockOp : targetOp.getOps<LockOp>()) {
    TileOp tileOp = lockOp.getTileOp();
    Coord coord = getCoord(tileOp);
    TileSummary &summary = summaries[coord];
    std::string lockName = getLockName(lockOp);
    summary.locks.push_back(lockName);
    recordResourceGroup(groups, lockName, coord, /*isBuffer=*/false);
    locks.push_back(buildLockJSON(lockOp));
  }
  for (MemOp memOp : targetOp.getOps<MemOp>()) {
    DMAContainerSummary dmaSummary = buildDMAContainerJSON(memOp, "mem");
    recordDMAGroupMembership(groups, dmaSummary);
    dmaSummariesByID.insert({dmaSummary.id, dmaSummary});
    dmas.push_back(std::move(dmaSummary.json));
  }
  for (MemTileDMAOp memTileDMAOp : targetOp.getOps<MemTileDMAOp>()) {
    DMAContainerSummary dmaSummary = buildDMAContainerJSON(memTileDMAOp, "memtile");
    recordDMAGroupMembership(groups, dmaSummary);
    dmaSummariesByID.insert({dmaSummary.id, dmaSummary});
    dmas.push_back(std::move(dmaSummary.json));
  }
  for (ShimDMAOp shimDMAOp : targetOp.getOps<ShimDMAOp>()) {
    DMAContainerSummary dmaSummary = buildDMAContainerJSON(shimDMAOp, "shim");
    recordDMAGroupMembership(groups, dmaSummary);
    dmaSummariesByID.insert({dmaSummary.id, dmaSummary});
    dmas.push_back(std::move(dmaSummary.json));
  }
  int packetFlowIndex = 0;
  for (PacketFlowOp packetFlowOp : targetOp.getOps<PacketFlowOp>()) {
    std::string packetFlowID = getPacketFlowID(packetFlowOp, packetFlowIndex);
    GroupSummary &group = getOrCreateGroup(
        groups, "packet_flow",
        llvm::formatv("packet_{0}_{1}", packetFlowOp.IDInt(), packetFlowIndex)
            .str());
    Region &region = packetFlowOp.getPorts();
    for (Operation &op : region.front()) {
      if (auto packetSourceOp = dyn_cast<PacketSourceOp>(op)) {
        TileOp tileOp = cast<TileOp>(packetSourceOp.getTile().getDefiningOp());
        group.tiles.insert(getCoord(tileOp));
      } else if (auto packetDestOp = dyn_cast<PacketDestOp>(op)) {
        TileOp tileOp = cast<TileOp>(packetDestOp.getTile().getDefiningOp());
        group.tiles.insert(getCoord(tileOp));
      }
    }
    group.packetFlows.insert(packetFlowID);
    packetFlowGroupIDs.push_back(getGroupID(group));
    packetFlows.push_back(buildPacketFlowJSON(packetFlowOp, packetFlowIndex++));
  }

  std::string routingJSONText;
  llvm::raw_string_ostream routingJSONStream(routingJSONText);
  if (failed(AIEFlowsToJSON(module, routingJSONStream, deviceName)))
    return failure();
  routingJSONStream.flush();

  auto parsedRouting = llvm::json::parse(routingJSONText);
  if (!parsedRouting) {
    module.emitOpError("failed to parse legacy routing JSON");
    return failure();
  }
  auto *routingObject = parsedRouting->getAsObject();
  if (!routingObject) {
    module.emitOpError("legacy routing JSON is not an object");
    return failure();
  }

  llvm::json::Array switchboxes;
  llvm::json::Array streams;
  int64_t totalPathLength = 0;
  std::map<std::string, llvm::json::Value> routesByID;

  for (const auto &entry : *routingObject) {
    llvm::StringRef key = entry.first;
    const llvm::json::Value &value = entry.second;

    if (key == "total_path_length") {
      if (auto integer = value.getAsInteger())
        totalPathLength = *integer;
      continue;
    }
    if (key == "route_all" || key == "end json")
      continue;

    if (key.starts_with("switchbox")) {
      auto *object = value.getAsObject();
      if (!object)
        continue;
      llvm::json::Object switchbox;
      switchbox["id"] = key.str();
      for (const auto &field : *object)
        switchbox[field.first] = field.second;
      switchboxes.push_back(std::move(switchbox));
      continue;
    }

    if (key.starts_with("route")) {
      routesByID.insert_or_assign(key.str(), value);
      continue;
    }
  }

  std::set<SourceKey> seenSources;
  std::vector<SourceKey> orderedCircuitSources;
  std::map<SourceKey, SmallVector<FlowOp, 4>> circuitFlowsBySource;
  int flowIndex = 0;
  for (FlowOp flowOp : targetOp.getOps<FlowOp>()) {
    TileOp sourceTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    SourceKey key = getSourceKey(sourceTile, flowOp.getSourceBundle(),
                                 flowOp.getSourceChannel());
    if (circuitFlowsBySource[key].empty())
      orderedCircuitSources.push_back(key);
    circuitFlowsBySource[key].push_back(flowOp);
  }
  for (const SourceKey &key : orderedCircuitSources) {
    std::string routeID = getStreamID(flowIndex++);
    auto routeIt = routesByID.find(routeID);
    if (routeIt == routesByID.end())
      continue;
    ArrayRef<FlowOp> flowOps = circuitFlowsBySource[key];
    FlowOp firstFlowOp = flowOps.front();
    TileOp sourceTile = cast<TileOp>(firstFlowOp.getSource().getDefiningOp());
    std::vector<GroupMatchSummary> groupMatches =
        collectCircuitGroupMatches(groups, dmaSummariesByID, sourceTile, flowOps);
    std::vector<std::string> groupIDs =
        selectBestCircuitGroupCandidates(groupMatches);
    bool usedFallbackFlowGroup = false;
    if (groupIDs.empty()) {
      GroupSummary &group =
          getOrCreateGroup(groups, "flow",
                           getFlowGroupName(sourceTile, firstFlowOp.getSourceBundle(),
                                            firstFlowOp.getSourceChannel()));
      group.tiles.insert(getCoord(sourceTile));
      for (FlowOp flowOp : flowOps) {
        TileOp destTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
        group.tiles.insert(getCoord(destTile));
      }
      groupIDs.push_back(getGroupID(group));
      usedFallbackFlowGroup = true;
    }
    CircuitProvenanceSummary provenance =
        summarizeCircuitProvenance(groupIDs, groupMatches, usedFallbackFlowGroup);
    recordStreamGroupMembership(groups, groupIDs, routeID);
    streams.push_back(buildCircuitStreamJSON(
        routeID, sourceTile, firstFlowOp.getSourceBundle(),
        firstFlowOp.getSourceChannel(), flowOps, routeIt->second, groupIDs,
        groupMatches, provenance));
    seenSources.insert(key);
  }

  int packetIndex = 0;
  for (PacketFlowOp packetFlowOp : targetOp.getOps<PacketFlowOp>()) {
    TileOp sourceTile;
    Port sourcePort;
    for (Operation &op : packetFlowOp.getPorts().front()) {
      if (auto packetSourceOp = dyn_cast<PacketSourceOp>(op)) {
        sourceTile = cast<TileOp>(packetSourceOp.getTile().getDefiningOp());
        sourcePort = packetSourceOp.port();
        break;
      }
    }
    SourceKey key =
        getSourceKey(sourceTile, sourcePort.bundle, sourcePort.channel);
    if (!seenSources.insert(key).second) {
      ++packetIndex;
      continue;
    }
    std::string routeID = getStreamID(flowIndex++);
    auto routeIt = routesByID.find(routeID);
    if (routeIt == routesByID.end()) {
      ++packetIndex;
      continue;
    }
    llvm::StringRef groupID = packetFlowGroupIDs[packetIndex];
    SmallVector<std::string, 1> groupIDs{groupID.str()};
    recordStreamGroupMembership(groups, groupIDs, routeID);
    streams.push_back(buildPacketStreamJSON(routeID, packetFlowOp, packetIndex,
                                            routeIt->second, groupID));
    ++packetIndex;
  }

  llvm::json::Object deviceJSON;
  deviceJSON["kind"] = stringifyAIEDevice(targetOp.getDevice()).str();
  deviceJSON["columns"] = model.columns();
  deviceJSON["rows"] = model.rows();

  llvm::json::Array tiles;
  for (int row = 0; row < model.rows(); ++row) {
    for (int col = 0; col < model.columns(); ++col) {
      Coord coord{col, row};
      const TileSummary &summary = summaries[coord];
      llvm::json::Object tile;
      tile["col"] = col;
      tile["row"] = row;
      tile["kind"] = getTileKind(model, col, row);
      tile["used"] = summary.declared || summary.corePresent || summary.memPresent ||
                     summary.switchboxPresent || summary.shimMuxPresent ||
                     summary.memTileDmaPresent || summary.shimDmaPresent ||
                     !summary.buffers.empty() || !summary.locks.empty();
      tile["core_present"] = summary.corePresent;
      tile["mem_present"] = summary.memPresent;
      tile["switchbox_present"] = summary.switchboxPresent;
      tile["shim_mux_present"] = summary.shimMuxPresent;
      tile["memtile_dma_present"] = summary.memTileDmaPresent;
      tile["shim_dma_present"] = summary.shimDmaPresent;
      tile["core_ids"] = toStringArray(summary.coreIDs);
      tile["buffers"] = toStringArray(summary.buffers);
      tile["locks"] = toStringArray(summary.locks);
      tiles.push_back(std::move(tile));
    }
  }

  llvm::json::Object metadataJSON;
  metadataJSON["flow_count"] = static_cast<int64_t>(streams.size());
  metadataJSON["switchbox_count"] = static_cast<int64_t>(switchboxes.size());
  metadataJSON["group_count"] = static_cast<int64_t>(groups.size());
  metadataJSON["total_path_length"] = totalPathLength;

  llvm::json::Object root;
  root["schema_version"] = 4;
  root["device"] = std::move(deviceJSON);
  root["tiles"] = std::move(tiles);
  root["cores"] = std::move(cores);
  root["buffers"] = std::move(buffers);
  root["locks"] = std::move(locks);
  root["dmas"] = std::move(dmas);
  root["switchboxes"] = std::move(switchboxes);
  root["streams"] = std::move(streams);
  root["packet_flows"] = std::move(packetFlows);
  root["groups"] = buildGroupJSON(groups);
  root["metadata"] = std::move(metadataJSON);

  llvm::json::Value topv(std::move(root));
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << llvm::formatv("{0:2}", topv) << "\n";
  output << ss.str();
  return success();
}

} // namespace xilinx::AIE
