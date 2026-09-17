cmake_minimum_required(VERSION 3.16)
project(optics_ingest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Werror)
endif()

# nlohmann/json: use a locally installed copy when available, otherwise fetch the
# pinned tag v3.11.3.
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# Statistics + window logic, deliberately free of JSON so it is unit-testable.
add_library(optics_window STATIC src/window.cpp)
target_include_directories(optics_window PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(optics_core STATIC src/jsonl_server.cpp src/validate.cpp)
target_include_directories(optics_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(optics_core PUBLIC optics_window nlohmann_json::nlohmann_json)

add_executable(optics_ingest src/main.cpp)
target_link_libraries(optics_ingest PRIVATE optics_core)

enable_testing()
add_executable(test_window tests/test_window.cpp)
target_link_libraries(test_window PRIVATE optics_window)
add_test(NAME test_window COMMAND test_window)
