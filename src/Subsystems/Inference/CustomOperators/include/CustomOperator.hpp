#pragma once

//

#include "onnxruntime_cxx_api.h"

//

struct CustomKernel {};

struct CustomOperator : public Ort::CustomOpBase<CustomOperator, CustomKernel> {
  constexpr char *getName() const {
    return "CustomOperator";
  }
};
