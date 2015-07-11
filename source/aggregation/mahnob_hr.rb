require './init'

sessions = MahnobSession.load(2..4000)

Pipeline.new(
    :output_dir => 'mahnob-hr',
    :normalize => false,
    :flow => {[HeartRateProcessor] => [AverageAggregator, RmssdAggregator]}
).run(sessions)

