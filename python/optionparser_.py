#custom option parser

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("num", 
                    help = "Enter n for the nth fibonacci number", 
                    type = int
                    )
args = parser.parse_args()



if __name__ == "__main__":
    print("This is a test session")
    exit(0)
