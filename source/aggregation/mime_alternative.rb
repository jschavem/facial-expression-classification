require './init'

sessions = MimeSession.load

Pipeline.new(
    :output_dir => 'mime-au-counts',
    :normalize => false,
    :flow => {[ActionUnitProcessor] => AverageAggregator}
).run(sessions)