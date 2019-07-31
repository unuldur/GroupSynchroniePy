session = 1
colors = ['y', 'b', 'r']
if session == 1:
    filenames = ["../data/session1/yellow.csv", "../data/session1/blue.csv", "../data/session1/red.csv"]
    offset = [0, 0, 224]
    start_groups = 0
    end_groups = -3
    groups = ["1", "2", "3"]
    max_len = 1600
    dec_correlattion = 8
if session == 2:
    filenames = ["../data/session2/yellow.csv", "../data/session2/blue.csv", "../data/session2/red.csv"]
    offset = [0, 0, 185]
    start_groups = 2
    end_groups = -3
    groups = ["3", "4", "5"]
    max_len = 1200
    dec_correlattion = 2
if session == 3:
    filenames = ["../data/session3/yellow.csv", "../data/session3/blue2.csv", "../data/session3/red.csv"]
    offset = [0, 0, 110]
    start_groups = 2
    end_groups = -4
    groups = ["5", "6", "7"]
    max_len = 1105
    dec_correlattion = 0
if session == 4:
    filenames = ["../data/session4Clean/yellow.csv", "../data/session4Clean/blue.csv", "../data/session4Clean/red.csv"]
    offset = [0, 5, 128]
    start_groups = 1
    end_groups = -4
    groups = ["7", "8", "9"]
    max_len = 1221
    dec_correlattion = 0
if session == 12:
    filenames = ["../data/session1/yellow.csv", "../data/session1/blue.csv", "../data/session1/red.csv", "../data/full_yellow.csv"]
    offset = [0, 0, 224, -2400]
    start_groups = 0
    end_groups = -3
    groups = ["1", "2", "3"]
    max_len = 1600
    dec_correlattion = 8

if session == 13:
    filenames = ["../data/session1/yellow.csv", "../data/session1/blue.csv", "../data/session1/red.csv", "../data/session2/blue.csv", "../data/session3/red.csv", "../data/session4/yellow.csv"]
    offset = [0, 0, 224,0 ,0, 100]
    start_groups = 0
    end_groups = -3
    groups = ["1", "2", "3"]
    max_len = 1600
    dec_correlattion = 8

length_frame = 3
frame_time = 0.01
size_frame = 0.025
feature_list = ["pauseTime", "turnPauseRatio", "silence", "turnDuration", "speakTime",
                "overlap", "interruptionOverlap", "interruption", "failedInterruptionOverlap",
                "failedInterruption"]
cohesion_file = "../data/responses.csv"
