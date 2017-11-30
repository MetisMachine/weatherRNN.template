import os
from skafossdk import *

print("Hello, world!")

skafos = Skafos()

res = skafos.engine.create_view(
    "weather_table", {"keyspace": "weather",
                      "table":
                      "weather_forecast"}, DataSourceType.Cassandra).result()
print("CREATED SPARK WEATHER TEMP TABLE")
print(res)

results = skafos.engine.query("SELECT * from weather_table LIMIT 5").result()
print("SELECTING FROM TABLE")
print(results)

schema = {
    "table_name":
    "my_table",
    "options": {
        "primary_key": ["column1", "column3"],
        "order_by": ["column3 desc"]
    },
    "columns": [{
        "column1": "varint"
    }, {
        "column2": "varchar"
    }, {
        "column3": "varchar"
    }, {
        "column4": "varchar"
    }]
}

data = [[1, "113", "123", "133"], [2, "213", "223", "233"]]

print("SAVING DATA TO CASSANDRA")
dataresult = skafos.engine.save(schema, data).result()
print(dataresult)

sparkview = skafos.engine.create_view("my_table", {"table": "my_table"},
                                      DataSourceType.Cassandra).result()
print("CREATED PROJECT SPARK TEMP TABLE")
print(sparkview)

mytableresults = skafos.engine.query("SELECT * from my_table LIMIT 5").result()
print("SELECTING FROM PROJECT KEYSPACE SPARK TABLE")
print(mytableresults)

insertresults = skafos.engine.query(
    "INSERT INTO TABLE my_table select 3, '313', '323', '333'").result()
print("INSERTED NEW RESULTS INTO SPARK TABLE")
print(insertresults)

mytableresultsafter = skafos.engine.query(
    "SELECT * from my_table LIMIT 5").result()
print("SELECTING FROM PROJECT KEYSPACE SPARK TABLE AFTER INSERT")
print(mytableresultsafter)

data = [[1, "updated1", "123", "133"], [3, "313", "323", "333"]]
print("UPDATING DATA TO CASSANDRA")
updateresult = skafos.engine.save(schema, data).result()
print(updateresult)

print("FINISHED ")
