import argparse # for command line arguments
import sys # only used for sys.stdout.write()

def main():
    # This defines what arguments the program accepts
    # and read the values from command line
    parser = argparse.ArgumentParser(description="A simple calculator")

    # type = float means that argument "10" in cli 
    # will be converted to number 10.0
    parser.add_argument("--x", type = float, default = 1.0, 
                        help = "What is the first number?")
    
    parser.add_argument("--y", type = float, default = 1.0, 
                        help = "What is the second number?")
    
    parser.add_argument("--operation", type = str, default = "add",
                        help = "What operation to perform? (add, sub, mul, div)")
    
    # this reads the arguments from command line
    args = parser.parse_args() 

    # this prints the result of calculation to the terminal
    sys.stdout.write("".join([str(calc(args.x, args.y, args.operation)), "\n"]))
    
def calc(x, y, operation):
    if operation == "add":
        return x + y
    elif operation == "sub":
        return x - y
    elif operation == "mul":
        return x * y
    elif operation == "div":
        return x / y

# only runs if YOU run calc.py directly  
# if you import calc.py in another file, this will not run  
if __name__ == '__main__':
    main()