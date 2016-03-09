#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

/* Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
 * Restrictions are listed in the LICENSE file distributed with this package. */

/* Return 1 if /dev/habis<d> can be opened for reading and writing, 0 otherwise. */
int checkdev(int d) {
	int fdes;
	char devname[32];
	const char * const devtmpl = "/dev/habis%d";

	if (snprintf(devname, 32, devtmpl, d) >= 32) {
		fprintf(stderr, "ERROR: Device name is too long\n");
		return 0;
	}

	fdes = open(devname, O_RDWR);

	if (fdes < 0) return 0;

	close(fdes);
	return 1;
}


int main (int argc, char **argv) {
	const int maxdevs = 8;
	int devct;

	FILE *rfile;
	const char * const devpath = "/sys/bus/pci/rescan";

	if (argc > 1) {
		fprintf(stderr, "USAGE: %s\n", argv[0]);
		return EXIT_FAILURE;
	}

	printf("Will rescan PCI device at path %s...\n", devpath);

	if (!(rfile = fopen(devpath, "w"))) {
		fprintf(stderr, "ERROR: Could not open %s for writing\n", devpath);
		return EXIT_FAILURE;
	}

	fprintf(rfile, "1");
	fclose(rfile);

	for (devct = 0; devct < maxdevs; ++devct) {
		if (!checkdev(devct)) break;
	}

	printf("%d valid HABIS devices\n", devct);

	return EXIT_SUCCESS;
}
