import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required= True,
                help = "name of the user")
args = vars(ap.parse_args())

#display a friendly message to the user
print("Hi therw {}, nice to meet you! you are cool".format(args["name"]))
