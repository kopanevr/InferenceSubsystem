/**
 * @brief
 *
 */

#pragma once

//

#include <thread>
#include <memory>

//


#include "InferenceContext.hpp"

//

#include "Subsystem.hpp"

//

// Подсистемы.

#include "Logger.hpp"

//

#include "onnxruntime_cxx_api.h"

//

namespace subsystemManager {
  class SubsystemManager;
}

//

namespace inference {
/// @brief
class Inference : public Subsystem {
public:
  /// @brief Деструктор.
  ~Inference();

  /// @brief
  /// @return
  static Inference *getInstance() {
    return instance_;
  }

private:
  /// @brief Конструктор.
  Inference() {
    // Инициализация.
    init();
  }

  Inference &operator=(const Inference &) = delete;
  Inference(const Inference &) = delete;

  /// @brief Дружественный класс.
  friend class subsystemManager::SubsystemManager;

  /// @brief Инициализация подсистемы.
  void init() override {
    SET_SUBSYSTEM_ID(subsystemManager::SubsystemId::Inference);
    SET_SUBSYSTEM_NAME("Inference");
  }

  /// @brief Предварительная настройка перед запуском подсистемы.
  bool setBeforeStartUp() override;
  /// @brief Предварительная настройка перед остановкой подсистемы.
  void setBeforeShutDown() override {}

  /// @brief Тело процесса.
  void processBody() override;

  /// @brief Конвейер.
  void pipeline();

  /// @brief
  void run();

  /// @brief
  bool body();

  /// @brief Подготовка входных тензоров.
  bool prepareInputTensors();
  /// @brief
  bool inference();
  /// @brief Подготовка выходных тензоров.
  bool prepareOutputTensors();

  /// @brief Устанавливает путь к модели.
  [[deprecated]] bool setModelFilePath();

  /// @brief
  [[deprecated]] void setRawBuffersSize();

  /// @brief Подготовка перед запуском вывода.
  /// @param options Опции. Дополнительно смотреть @ref inference::prepareSettings.
  bool prepareBeforeStartInference([[maybe_unused]] const uint8_t options = 0);
  /// @brief Подготовка провайдера вывода.
  /// @param options Опции. Дополнительно смотреть @ref inference::prepareSettings.
  bool prepareProvider([[maybe_unused]] const uint8_t options = 0);
  /// @brief Создание входных и выходных тензоров.
  /// @param
  bool createInputOutputTensors();
  /// @brief Возвращает информацию о модели.
  /// @param inferenceContext Контекст вывода.
  /// @return Информация о модели.
  [[nodiscard]] std::unique_ptr<ModelInfo> getModelInfo(InferenceContext &inferenceContext);

private:
  /// @brief
  static inline Inference *instance_;

  /// @brief
  std::thread inferenceThread_;

  /// @brief Контекст вывода.
  std::unique_ptr<InferenceContext> inferenceContext_;
};

/// @brief Устанавливает путь к модели.
inline bool Inference::setModelFilePath() {
  if(!inferenceContext_) {
    return false;
  }

  inferenceContext_->modelPath.modelDirectoryPath = inference::modelDirectoryPath;
  inferenceContext_->modelPath.modelFileName = inference::modelFileName;

  inferenceContext_->optimizedModelPath.modelDirectoryPath = inference::optimizedModelDirectoryPath;
  inferenceContext_->optimizedModelPath.modelFileName = inference::optimizedModelFileName;

  return true;
}

/// @brief Устанавливает размеры буферов для входного и выходного тензоров.
[[deprecated]] inline void Inference::setRawBuffersSize() {
  auto resizeBuffer = [this](const std::unique_ptr<TensorInfo> &tensorInfo, std::unique_ptr<Tensor> &tensor) -> size_t {
    const auto &shape = tensorInfo->shape;
    if (shape->empty()) {
      return {};
    }

    size_t totalElements = (size_t)1;

    for (const auto &dim : *shape) {
      totalElements *= static_cast<size_t>(dim);
    }

    const auto &inputTensorElementDataType = tensorInfo->tensorElementDataType;
    size_t elementSize = sizeof(float);

    if (inputTensorElementDataType == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      elementSize = sizeof(float);
    }

    const size_t bufferSize = totalElements * elementSize;
    tensor->rawData.resize(bufferSize);

    return bufferSize;
  };

  INFO(
    "Размер буфера входного тензора: ",
    resizeBuffer(
      inferenceContext_->modelInfo->inputTensorInfo,
      inferenceContext_->inputTensor),
    " [байт]."
  );
  INFO(
    "Размер буфера выходного тензора: ",
    resizeBuffer(
      inferenceContext_->modelInfo->outputTensorInfo,
      inferenceContext_->outputTensor),
    " [байт]."
  );
}
} // namespace inference
