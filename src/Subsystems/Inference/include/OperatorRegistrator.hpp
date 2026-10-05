#pragma once

//

#include <memory>

//

#include "onnxruntime_cxx_api.h"

//

/// @brief
class OperatorRegistrator final {
public:
  OperatorRegistrator(std::unique_ptr<Ort::SessionOptions> sessingOptions)
      : sessingOptions_(std::move(sessingOptions)) {}

  ~OperatorRegistrator();


  [[nodiscard]] std::unique_ptr<Ort::SessionOptions> registerCustomOpt() {
    return {};
  }

private:
  std::unique_ptr<Ort::SessionOptions> sessingOptions_;
};
