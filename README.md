# VisualizeDecisionTree
This project creates a visual **Decision Tree** based on a dataset.<br/><br/>


## Instalations

To run this project, you must install the following packages:

- pandas
- numpy
- pydot (for visualizing)

install pandas:
```
pip install pandas
```
By installing Pandas, Numpy will also be installed in your system.

install pydot:
```
pip install pydot
```

## Usage

After running the program, a **Result** directory is created in its current directory and two directories are created within the result:
- **VisualizeDecisionTree**: In this dir, for each tutorial, the decision tree is saved as an image using the [pydot](https://github.com/pydot/pydot) package.
  <br/>example:<br/>
  <br/>![visual DT](/Results/VisualizeDecisionTrees/monks_t40_decision_tree.png)
  
- **Accuracy**: Evaluates the test dataset using the decision tree constructed with each training set and stores the accuracy in a [csv](https://en.wikipedia.org/wiki/Comma-separated_values) file.
    <br/>example:<br/>
  <br/>![csv example](/Results/Accuracy/CSVExampleForReadMe.png)
  
