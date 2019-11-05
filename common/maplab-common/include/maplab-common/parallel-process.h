#ifndef MAPLAB_COMMON_PARALLEL_PROCESS_H_
#define MAPLAB_COMMON_PARALLEL_PROCESS_H_
#include <cmath>
#include <thread>  // NOLINT
#include <vector>

#include <glog/logging.h>
#include <maplab-common/threading-helpers.h>

// This is a helper to call a user provided functor or lamda with block indices
// in a threaded context.
//
// First create your functor:
// struct Squarer {
//   Squarer(const std::vector<double>& input, std::vector<double>* output)
//       : input_(input),
//         output_(CHECK_NOTNULL(output)) {
//   }
//   const std::vector<double>& input_;
//   std::vector<double>* output_;
//   void operator()(const std::vector<size_t>& range) const {
//     for (size_t i : range) {
//       (*output_)[i] = input_[i] * input_[i];
//     }
//   }
// };
//
// Now you can run this in parallel:
// std::vector<double> data, results;
// data.resize(10, 7);
// results.resize(10);
//
// Squarer squarer(data, &results);
// ParallelProcess(data.size(), squarer, true, 16);

namespace common {

    /*
     * std::function<void(const std::vector<size_t>)> serialize_function =
            [&](const std::vector<size_t>& range)
            {
                size_t num_processed_tasks = 0u;
                progress_bar.setNumElements(2u * range.size());
                proto::VIMap proto;
                for (size_t task_idx : range)
                {
                    proto.Clear();
                    switch (task_idx)
                    {
                        case internal::kProtoListMissionsIndex:
                            serializeMissionsAndBaseframes(map, &proto);
                            break;
                        case internal::kProtoListEdgesIndex:
                            serializeEdges(map, &proto);
                            break;
                        case internal::kProtoListLandmarkIndexIndex:
                            serializeLandmarkIndex(map, &proto);
                            break;
                        case internal::kProtoListOptionalSensorData:
                            serializeOptionalSensorData(map, &proto);
                            break;
                        default: {
                            // Case vertices.
                            serializeVertices(
                                    map, (task_idx - internal::kProtoListVerticesStartIndex) *
                                         save_config.vertices_per_proto_file,
                                    save_config.vertices_per_proto_file, &proto);
                            break;
                        }
                    }
                    progress_bar.update(++num_processed_tasks);

                    CHECK(function(task_idx, proto));
                    progress_bar.update(++num_processed_tasks);
                }
            };


            constexpr size_t kProtoListMissionsIndex = 0u;
constexpr size_t kProtoListEdgesIndex = 1u;
constexpr size_t kProtoListLandmarkIndexIndex = 2u;
constexpr size_t kProtoListOptionalSensorData = 3u;
     */




// Usually batches which are too small are not threaded. Set
// "always_parallelize" to true to force threading even if every thread only
// gets a single item. The processes will execute data indices in the range of
// [start_idx, end_idx)
template <typename Functor>
void ParallelProcess(
    const size_t start_index, const size_t end_index, const Functor& functor,
    const bool always_parallelize, const size_t num_threads)
    {
        CHECK_GE(start_index, 0u) << "Start index needs to be >= 0.";
        CHECK_GT(end_index, start_index)
            << "End index needs to be bigger than the start index.";
        std::vector<std::vector<size_t> > blocks;

        const size_t num_items = end_index - start_index;//项目个数
        size_t num_items_per_block = num_items;
        if (num_items < num_threads * 2 && !always_parallelize) //因为always_parallelize被设成了true，所以不会去调用这个
        {
            blocks.resize(1);
        } else
            {
            num_items_per_block = std::ceil(//向上取整，就是每个线程要分配多少个项目
                    static_cast<double>(num_items) / static_cast<double>(num_threads));
            const int num_blocks = std::ceil(//向上取整，总共要多少个块
                    static_cast<double>(num_items) /
                    static_cast<double>(num_items_per_block));
            blocks.resize(num_blocks);
        }

        size_t data_index = start_index;
        std::vector<std::thread> threads;
        for (size_t block_idx = 0u; block_idx < blocks.size(); ++block_idx)
        {
            std::vector<size_t>& block = blocks[block_idx];
            for (size_t item_idx = 0u;
                 (item_idx < num_items_per_block) && (data_index < end_index);
                 ++item_idx)
            {
                CHECK_GE(data_index, start_index);
                CHECK_LT(data_index, end_index);
                block.push_back(data_index);//block是当前这个线程所有需要保存的项目集合
                ++data_index;
            }
            threads.push_back(//初始化函数
                    std::thread([&functor, &block]() -> void { functor(block); }));
        }

        CHECK_EQ(threads.size(), blocks.size());
        for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
            threads[block_idx].join();
        }
    }

// Usually batches which are too small are not threaded. Set
// "always_parallelize" to true to force threading even if every thread only
// gets a single item.
template <typename Functor>
void ParallelProcess(
    const size_t num_items, const Functor& functor,
    const bool always_parallelize, const size_t num_threads)
    {
  CHECK_GE(num_items, 0u);
  CHECK_GT(num_threads, 0u) << "Num threads must be larger than 0.";

  if (num_items == 0u) {
    // Nothing to do here.
    return;
  }

  constexpr size_t kStartIndex = 0u;
  const size_t end_index = num_items;//压缩包的个数
  ParallelProcess(
      kStartIndex, end_index, functor, always_parallelize, num_threads);
}

}  // namespace common
#endif  // MAPLAB_COMMON_PARALLEL_PROCESS_H_
