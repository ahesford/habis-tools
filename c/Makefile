CC=gcc
LD=$(CC)

OBJS= habis-remove.o habis-rescan.o

all: habis-remove habis-rescan

habis-remove: habis-remove.o
	$(LD) $(LFLAGS) -o $@ habis-remove.o $(LIBDIR) $(LIBS)

habis-rescan: habis-rescan.o
	$(LD) $(LFLAGS) -o $@ habis-rescan.o $(LIBDIR) $(LIBS)

clean:
	rm -f $(OBJS) habis-remove habis-rescan
	rm -f *.core core 

.SUFFIXES: .o .c

.c.o:
	$(CC) $(DFLAGS) $(CFLAGS) -o $@ -c $<
