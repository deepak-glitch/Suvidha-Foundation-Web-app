var video = document. getElementById('imp-in');
var source = document. getElementById('source');
video_path= "static/input-videos/input-videos-" + videofile.filename
source. setAttribute('src', video_path);
video. load();
video. play();