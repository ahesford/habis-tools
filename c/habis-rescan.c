#include <stdio.h>
#include <stdlib.h>

/* Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
 * Restrictions are listed in the LICENSE file distributed with this package. */

int main (int argc, char **argv) {
	char *devpath = "/sys/bus/pci/rescan";
	FILE *rfile;

	if (argc > 1) {
		fprintf(stderr, "USAGE: %s\n", argv[0]);
		return EXIT_FAILURE;
	}

	printf("Will rescan PCI device at path %s\n", devpath);

	if (!(rfile = fopen(devpath, "w"))) {
		fprintf(stderr, "ERROR: Could not open %s for writing\n", devpath);
		return EXIT_FAILURE;
	}

	fprintf(rfile, "1");
	fclose(rfile);

	return EXIT_SUCCESS;
}