require './init'

sessions = MahnobSession.load(2..4000)

Pipeline.new(
    :output_dir => 'mahnob-au-test',
    :normalize => false,
    :flow => {[EmfacsActionUnitProcessor] => []}
).run(sessions)