#!/usr/bin/env python

from twisted.spread import pb
from twisted.internet import reactor

from habis import wrappers
from habis.conductor import HabisConductor

def main():
	HabisConductor.registerWrapper("echo", wrappers.Echo)
	HabisConductor.registerWrapper("test256", wrappers.Test256)
	port = reactor.listenTCP(8088, pb.PBServerFactory(HabisConductor()))
	print 'Listening on %s' % port.getHost()
	reactor.run()


if __name__ == "__main__":
	main()
