
<div align="center">
  <img src="https://cgnal.com/wp-content/uploads/2016/05/Gnal_logo.png"><br>
</div>

-----------------

CGnal core
====

[![PyPI](https://img.shields.io/pypi/v/cgnal-core.svg)](https://pypi.python.org/pypi/cgnal-core)
[![PyPI version](https://img.shields.io/pypi/pyversions/cgnal-core.svg)](https://pypi.python.org/pypi/cgnal-core)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://cgnal.github.io/cgnal-core/)
![Python package](https://github.com/CGnal/cgnal-core/workflows/CI%20-%20Build%20and%20Test/badge.svg)

--------------------------------------------------------------------------------


A Python library defining data structures optimized for machine learning pipelines 


## What is it ?
**cgnal-core** is a Python package with modular design that provides powerful abstractions to build data 
ingestion pipelines and run end to end machine learning pipelines. 
The library offers lightweight object-oriented interface to MongoDB as well as Pandas based data structures. 
The aim of the library is to provide extensive support for developing machine learning based applications 
with a focus on practicing clean code and modular design. 

## Features
Some cool features that we are proud to mention are: 

### Data layers 
1. Archiver: Offers an object-oriented design to perform ETL on Mongodb collections as well as Pandas DataFrames.
2. DAO: Data Access Object to allow archivers to serialize domain objects into the proper persistence layer support 
object (e.g. in the case of MongoDB, a DAO serializes a domain object into a MongoDB document) and to parse objects
retrieved from the given persistence layer in the correct representation in our framework (e.g. a text will be parsed in 
a Document while tabular data will be parsed in a pandas DataFrame).
3. Database: Object representing a relational database
4. Table: Object representing a table of a relational database

### Data Model
Offers the following data structures: 
1. Document : Data structure specifically designed to work with NLP applications that parses a json-like document 
into a couple of uuid and dictionary of information.
2. Sample : Data structure representing an observation (a.k.a. sample) as used in machine learning applications
3. MultiFeatureSample : Data structure representing an observation defined by a nested list of arrays.
4. Dataset : Data structure designed to be used specifically for machine learning applications representing a collection 
of samples.

## Installation
From pypi server
```
pip install cgnal-core
```

From source
```
git clone https://github.com/CGnal/cgnal-core
cd cgnal-core
make install
```

## Tests 
```
make tests
```

## Checks 
To run predefined checks (unit-tests, linting checks, formatting checks and static typing checks):
```
make checks
```

## Examples 

#### Data Layers
Creating a Database of Table objects

```python
import pandas as pd
from cgnal.core.data.layer.pandas.databases import Database

# sample df
df1 = pd.DataFrame([[1, 2, 3], [6, 5, 4]], columns=['a', 'b', 'c'])

# creating a database 
db = Database('/path/to/db')
table1 = db.table('df1')

# write table to path
table1.write(df1)
# get path  
table1.filename

# convert to pandas dataframe 
table1.to_df()

# get table from database 
db.__getitem__('df1')
```

Using an Archiver with Dao objects

```python
from cgnal.core.data.layer.pandas.archivers import CsvArchiver
from cgnal.core.data.layer.pandas.dao import DataFrameDAO

# create a dao object 
dao = DataFrameDAO()

# create a csv archiver 
arch = CsvArchiver('/path/to/csvfile.csv', dao)

# get pandas dataframe 
arch.data

# retrieve a single document object 
doc = next(arch.retrieve())
# retrieve a list of document objects 
docs = [i for i in arch.retrieve()]
# retrieve a document by it's id 
arch.retrieveById(doc.uuid)

# archive a single document 
doc = next(self.a.retrieve())
# update column_name field of the document with the given value
doc.data.update({'column_name': value})
# archive the document 
arch.archiveOne(doc)
# archive list of documents
a.archiveMany([doc, doc])

# get a document object as a pandas series 
arch.dao.get(doc)
```
#### Data Model

Creating a PandasDataset object

```python
import pandas as pd
from cgnal.core.data.model.ml import PandasDataset

dataset = PandasDataset(features=pd.concat([pd.Series([1, np.nan, 2, 3], name="feat1"),
                                            pd.Series([1, 2, 3, 4], name="feat2")], axis=1),
                        labels=pd.Series([0, 0, 0, 1], name="Label"))

# access features as a pandas dataframe 
dataset.features
# access labels as pandas dataframe 
dataset.labels
# access features as a python dictionary 
dataset.getFeaturesAs('dict')
# access features as numpy array 
dataset.getFeaturesAs('array')

# indexing operations 
# access features and labels at the given index as a pandas dataframe  
dataset.loc(2).features
dataset.loc(2).labels
```

Creating a PandasTimeIndexedDataset object

```python
import pandas as pd
from cgnal.core.data.model.ml import PandasTimeIndexedDataset

dateStr = [str(x) for x in pd.date_range('2010-01-01', '2010-01-04')]
dataset = PandasTimeIndexedDataset(
    features=pd.concat([
        pd.Series([1, np.nan, 2, 3], index=dateStr, name="feat1"),
        pd.Series([1, 2, 3, 4], index=dateStr, name="feat2")
    ], axis=1))
```

## How to contribute ? 

We are very much willing to welcome any kind of contribution whether it is bug report, bug fixes, contributions to the 
existing codebase or improving the documentation. 

### Where to start ? 
Please look at the [Github issues tab](https://github.com/CGnal/cgnal-core/issues) to start working on open 
issues 

### Contributing to cgnal-core 
Please make sure the general guidelines for contributing to the code base are respected
1. [Fork](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) the cgnal-core repository. 
2. Create/choose an issue to work on in the [Github issues page](https://github.com/CGnal/cgnal-core/issues). 
3. [Create a new branch](https://docs.github.com/en/get-started/quickstart/github-flow) to work on the issue. 
4. Commit your changes and run the tests to make sure the changes do not break any test. 
5. Open a Pull Request on Github referencing the issue.
6. Once the PR is approved, the maintainers will merge it on the main branch.