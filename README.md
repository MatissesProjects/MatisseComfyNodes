So here are a few useful nodes I have needed while building different workflows.
Lets break them down

### List length:
gives you the length of the list of images added

### rgba video combine:
gives you a gif with an alpha layer applied, background remove images or segment them first for this to do much

### clip strings:
why have 2, we can have a single clip input and a positive and negitive prompt from one box

### file reader:
great for always reading the content of a file, so say you have a different application modifying it (like twitch chat) that can be used to directly influence different parts of the graph with this reader.

### string concat:
simple string concatenation

### string:
our own string primitive so we dont have to depend on other libraries for now

### image selector:
####BROKEN
selects an image from a list of images. Currently broken needs to select correct channel
