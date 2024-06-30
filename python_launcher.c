#include <stdio.h>
#include <windows.h>

int main(int argc, char *argv[]) {
    HMODULE hPython = LoadLibrary("python3.dll");
    if (!hPython) {
        fprintf(stderr, "Failed to load python3.dll\n");
        return 1;
    }

    int (*Py_Main)(int, wchar_t **);
    Py_Main = (int (*)(int, wchar_t **))GetProcAddress(hPython, "Py_Main");
    if (!Py_Main) {
        fprintf(stderr, "Failed to locate Py_Main in python3.dll\n");
        FreeLibrary(hPython);
        return 1;
    }

    // Convert char* args to wchar_t* args
    wchar_t **wargv = (wchar_t **)malloc((argc + 1) * sizeof(wchar_t *));
    for (int i = 0; i < argc; i++) {
        size_t len = strlen(argv[i]) + 1;
        wargv[i] = (wchar_t *)malloc(len * sizeof(wchar_t));
        mbstowcs(wargv[i], argv[i], len);
    }
    wargv[argc] = NULL;

    int result = Py_Main(argc, wargv);

    for (int i = 0; i < argc; i++) {
        free(wargv[i]);
    }
    free(wargv);

    FreeLibrary(hPython);
    return result;
}
