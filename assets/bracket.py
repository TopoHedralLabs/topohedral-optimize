
import scipy.optimize as opt
import json


def bracket_tests(results: dict):

    def f(x):
        return x**2 - 2

    lims = [(-1.0, 1.0), 
            (0.0, 1.0), 
            (-1.0, 0.0), 
            (100, 100.001), 
            (-100.001, -100)
    ]
    descs = [
        "Test of the bracket function for x^2-2 where a = -1 and b = 1",
        "Test of the bracket function for x^2-2 where a = 0 and b = 1",
        "Test of the bracket function for x^2-2 where a = -1 and b = 0",
        "Test of the bracket function for x^2-2 where a = 100 and b = 100.001",
        "Test of the bracket function for x^2-2 where a = -100.001 and b = -100"
    ]

    for i in range(len(lims)):
        desc = descs[i]
        a = lims[i][0]
        b = lims[i][1]
        x = opt.bracket(f, a, b)

        dataset_name = "bracket_test{}".format(i+1)
        results[dataset_name] = dict()
        results[dataset_name]["description"] = desc
        results[dataset_name]["values"] = dict()
        results[dataset_name]["values"] = dict()
        results[dataset_name]["values"]["a"] = a
        results[dataset_name]["values"]["b"] = b
        results[dataset_name]["values"]["results"] = x


def bracket_test5():

    def f(x): 
        return 10.0

    try:
        x = opt.bracket(f)
    except: 
        print('Exception occured')


def main():

    results = dict()
    bracket_tests(results)

    with open('bracket.json', 'w') as f:
        json.dump(results, f, indent=4)





if __name__ == '__main__':

    main()