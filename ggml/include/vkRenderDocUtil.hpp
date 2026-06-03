#ifndef _VKRENDERDOCUTIL_HPP
#define _VKRENDERDOCUTIL_HPP

#include "renderdoc_app.h"

#include <stdexcept>
#include <iostream>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
using LibraryHandle = HMODULE;
#define LoadLibraryFunc        LoadLibrary
#define GetSymbolFunc          GetProcAddress
#define UnloadLibraryFunc      FreeLibrary
#define RENDERDOC_LIBRARY_NAME "renderdoc.dll"
#else
#include <dlfcn.h>
using LibraryHandle = void*;
#define LoadLibraryFunc(name)  dlopen(name, RTLD_LAZY)
#define GetSymbolFunc          dlsym
#define UnloadLibraryFunc      dlclose
#define RENDERDOC_LIBRARY_NAME "renderdoc.so"
#endif

class DynamicLibrary
{
public:

    explicit DynamicLibrary(const std::string& libraryPath)
        : handle(nullptr)
    {
        handle = LoadLibraryFunc(libraryPath.c_str());
        if (!handle)
        {
            throw std::runtime_error("Failed to load library: " + libraryPath);
        }
    }

    ~DynamicLibrary()
    {
        if (handle)
        {
            UnloadLibraryFunc(handle);
        }
    }

    template <typename T> T getFunction(const std::string& funcName)
    {
        auto func = reinterpret_cast<T>(GetSymbolFunc(handle, funcName.c_str()));
        if (!func)
        {
            throw std::runtime_error("Failed to load function: " + funcName);
        }
        return func;
    }

private:

    LibraryHandle handle;
};

class RenderDocUtil
{
public:

    RenderDocUtil(void)
    {

        DynamicLibrary lib(RENDERDOC_LIBRARY_NAME);
        m_library                   = &lib;
        using FuncType              = void (*)();
        ::pRENDERDOC_GetAPI pGetApi = (::pRENDERDOC_GetAPI)m_library->getFunction<FuncType>("RENDERDOC_GetAPI");
        const int           ret     = pGetApi(eRENDERDOC_API_Version_1_1_2, (void**)&m_api);

        if (ret == 1)
        {
            m_api->TriggerCapture();

            m_valid = true;
        }
        else
        {
            std::cout << "RENDERDOC_GetAPI returned " << ret << " status, RenderDoc API not available" << std::endl;
        }
    }

    ~RenderDocUtil(void)
    {
    }

    bool isValid(void)
    {
        return m_valid;
    }

    void startFrame()
    {
        if (!isValid())
            return;
        m_api->StartFrameCapture(nullptr, nullptr);
    }
    void endFrame()
    {
        if (!isValid())
            return;
        m_api->EndFrameCapture(nullptr, nullptr);
    }

private:

    DynamicLibrary*        m_library;
    ::RENDERDOC_API_1_1_2* m_api;
    bool                   m_valid;
};

#endif // _VKRENDERDOCUTIL_HPP
