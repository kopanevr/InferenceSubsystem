/**
 * @brief
 *
 */

#pragma once

//

#include <memory>

//

#include "onnxruntime_cxx_api.h"

//

/// @brief
class OperatorsRegistrator final {
public:
  OperatorsRegistrator(std::unique_ptr<Ort::SessionOptions> &sessionOptions)
      : sessionOptions_(std::move(sessionOptions)) {}

  ~OperatorsRegistrator() = default;

  [[nodiscard]] std::unique_ptr<Ort::SessionOptions> registerCustomOpt() {
    if (!sessionOptions_) {
      return {};
    }

    return std::move(sessionOptions_);
  }

private:
  std::unique_ptr<Ort::SessionOptions> sessionOptions_;
};
