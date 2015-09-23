#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main (int argc, char **argv) {
	char *devfmt = "/sys/bus/pci/devices/%s/remove";
	char *defdevice = "0000:00:03.0";
	char devpath[PATH_MAX] = "";

	FILE *rfile;

	if (argc < 2) {
		snprintf(devpath, PATH_MAX, devfmt, defdevice);
	} else if (argc > 2) {
		fprintf(stderr, "USAGE: %s [device]\n", argv[0]);
		return EXIT_FAILURE;
	} else {
		snprintf(devpath, PATH_MAX, devfmt, argv[1]);
	}

	printf("Will remove PCI device at path %s\n", devpath);

	if (!(rfile = fopen(devpath, "w"))) {
		fprintf(stderr, "ERROR: Could not open %s for writing\n", devpath);
		return EXIT_FAILURE;
	}

	fprintf(rfile, "1");
	fclose(rfile);

	return EXIT_SUCCESS;
}
