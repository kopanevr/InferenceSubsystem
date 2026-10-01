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
} // namespace inference
