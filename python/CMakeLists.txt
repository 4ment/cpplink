# The Python extension: one module, `cpplink._cpplink`, over the static core.
# Built only under -DCPPLINK_BUILD_PYTHON=ON, which pyproject.toml sets; the
# CLI and the tests never see it.

find_package(pybind11 CONFIG REQUIRED)

file(GLOB CPPLINK_BINDING_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.hpp)

pybind11_add_module(_cpplink MODULE ${CPPLINK_BINDING_SOURCES})
target_link_libraries(_cpplink PRIVATE cpplink_core)
# Headers are named from `python/`, as the core's are from `src/`.
target_include_directories(_cpplink PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(_cpplink PRIVATE
    CPPLINK_VERSION_STRING="${PROJECT_VERSION}")

# The module links the conda Arrow the interpreter's pyarrow also loads, and has
# to find it from wherever pip puts the module: keep the link directories in the
# install rpath rather than relying on the environment.
set_target_properties(_cpplink PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON
    BUILD_WITH_INSTALL_RPATH OFF)

install(TARGETS _cpplink DESTINATION cpplink)
