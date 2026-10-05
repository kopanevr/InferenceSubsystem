#pragma once

class ApplicationPreaparator final {
public:
  /// @brief Подготовка при инициализации.
  /// @details
  /// @param argc Количество аргументов.
  /// @param argv Указатель на список аргументов.
  bool prepare(int argc, char *argv[]);

private:

};