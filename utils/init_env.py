import os
hostname = os.popen("hostname").read().split("\n")[0]
if(hostname != "reckoner1429-Predator-PH315-52"):
    from google.colab import drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    import sys
    sys.path.append('/content/gdrive/My Drive/video-emotion-recognition')
    print(sys.path)

    os.system('cp "/content/gdrive/My Drive/ssh/known_hosts" "/root/.ssh/known_hosts"')
    os.system('cp "/content/gdrive/My Drive/ssh/id_rsa" "/root/.ssh/id_rsa"')
    os.system('cp "/content/gdrive/My Drive/ssh/id_rsa.pub" "/root/.ssh/id_rsa.pub"')
    
    os.system('ls "/root/.ssh"')
    os.system('chmod 644 "/root/.ssh/known_hosts"')
    os.system('chmod 600 "/root/.ssh/id_rsa"')
    
    os.system('ssh -T git@github.com')
    
    os.system('pwd')
    
    if(os.path.getsize("'/content/gdrive/My Drive/github/video-emotion-recognition'") == 0):
        os.system('git clone "git@github.com:abhishekbisht1429/video-emotion-recognition.git" "gdrive/My Drive/github/video-emotion-recognition"')
    
    os.system("%cd '/content/gdrive/My Drive/github/video-emotion-recognition'")
    os.system("git pull origin master")
    
