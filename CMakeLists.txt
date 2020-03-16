cmake_minimum_required (VERSION 3.12)

# Set options for this project.
project(tests ${PROJECT_NAME})

set (PROJECT_SOURCE_DECLARATION_DIRECTORY ${PROJECT_SOURCE_DIR}/include)
set (PROJECT_SOURCE_DEFINITION_DIRECTORY ${PROJECT_SOURCE_DIR})
set (MAIN_FILE ${PROJECT_SOURCE_DEFINITION_DIRECTORY}/main.cpp)

# Set options used by "CMake".
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=18050")
# Set options used by "FindBoost.cmake".



find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS context coroutine date_time timer thread system program_options regex REQUIRED)

include_directories (${PROJECT_SOURCE_DECLARATION_DIRECTORY})


# "Add an executable to the project using the specified source files."
add_executable (${PROJECT_NAME} ${MAIN_FILE})


set_target_properties(${PROJECT_NAME}  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    COMPILE_OPTIONS "-Wpedantic;-Wall;-Wextra"
    coroutine PROPERTIES

)
# "Link a target to given libraries."
 target_link_libraries (${PROJECT_NAME} ${Boost_LIBRARIES})
 
