## Description   

Connects directly to the postgres database.  
Scans she database for images added recently   
Requests the vectors from the ML server for the keywords   
Checks to see if they match.   
Performs actions as required using the Immich API   
Adds the entry and action taken to a sqlite db to save rescan time.  



## How to use   

Given how it is written, the Postgres must be passwordless, this can be fixed if needed, but suits my config.    
Otherwise, should work for all installations, just copy env.example to .env and configure it as needed.  
Any albums you want to use, must be created in Immich, the script does not create the album, it searches for a matching name.   
All keywords, albums and actions are configured in config.json.   

I have worked with a couple people and their requirements to come up with the thresholds, and keywords used for Documents, and NSFW. 
Keywords and thresholds for screenshots is not working well though.  

Running debug_asset.py on a asset id will show what distance every keyword is from said asset.  

Configure it to run using cron, or some other scheduler.  


## Disclaimer   

I am not tech support, I am simply putting this out there incase anyone else wants it.   
It has only been tested on 2 installations, one manually set up on ubuntu, and one using nixos flakes.  


## Code Origin, Author and License 
This is not hand coded.  Well, a chunk of it is, but in reality, at least 3/4 was written by Gemini 2.5-pro-exp-03-25 as an agent in Cursor 0.48   
Therefor I feel anything less permissive than MIT Licence would be dishonest.   So MIT is is.
