#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

int main(int argc, char *argv[]) {
	if (argc != 4) {
		fprintf(stderr, "Usage: %s lib_name func_name arg\n", argv[0]); return 1;
	}

	const char *lib_name = argv[1];
	const char *func_name = argv[2];
	int arg = atoi(argv[3]);

	void *handle = dlopen(lib_name, RTLD_LAZY);
	if (!handle) {
		fprintf(stderr, "Error loading library: %s\n", dlerror());
		return 1;
	}

	int (*func)(int);
	*(void **) (&func) = dlsym(handle, func_name);
	if (!func) {
		fprintf(stderr, "Error getting function pointer: %s\n", dlerror());
		dlclose(handle);
		return 1;
	}

	int result = func(arg);
	printf("%d\n", result);

	dlclose(handle);
	return 0;
}
