import torchvision
video_path = "path to a test video"
# Constructor allocates memory and a threaded decoder
# instance per video. At the moment it takes two arguments:
# path to the video file, and a wanted stream.
reader = torchvision.io.VideoReader(video_path, "video")

# The information about the video can be retrieved using the
# `get_metadata()` method. It returns a dictionary for every stream, with
# duration and other relevant metadata (often frame rate)
reader_md = reader.get_metadata()

# metadata is structured as a dict of dicts with following structure
# {"stream_type": {"attribute": [attribute per stream]}}
#
# following would print out the list of frame rates for every present video stream
print(reader_md["video"]["fps"])

# we explicitly select the stream we would like to operate on. In
# the constructor we select a default video stream, but
# in practice, we can set whichever stream we would like
video.set_current_stream("video:0")
