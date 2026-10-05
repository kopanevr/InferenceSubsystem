/**
 * @file
 * @brief
 */

//

#include "Application.hpp"

//

/// @brief
/// @param argc
/// @param argv
/// @return
int main(int argc, char *argv[]) {
  auto *const app = app::Application::getInstance();
  const int ret = app->init(argc, argv);
  if (ret) {
    return ret;
  }

  return app->exec();
}
