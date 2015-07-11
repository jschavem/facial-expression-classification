require './init'

sessions = BinedSession.load

Pipeline.new(
    :output_dir => 'bined-koelstra-copy',
    :normalize => false,
    :debug => true,
    :processors => [EmfacsActionUnitProcessor],
    :aggregator => nil
).run(sessions)
