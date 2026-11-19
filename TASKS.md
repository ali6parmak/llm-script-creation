## Plan International

- Use the segmentations in the `segmentations` collection, to try to infer the different properties. The ones that appear in the segmentation from `entities` from the template `Document`.
  
```
 db.getCollection('entities').find({template: ObjectId("5bfbb1a0471dd0fc16ada146")}) 
```


## UPR

- Use the segmentation to,
  - Extract the recommendations
  - The state in the first page
  - The recommendation number
  - And the recommendation countries


What we want to know in the benchmarks we are going to do is that if an LLM can create scripts to perform these extractions. For the country extractions, we can pass the list of countries that Uwazi have, so it can know to create an heuristic to pick the right country. We can also pass the position of the segments and size, and we want to know if the performance improves, and try different prompts explaining the task so we know which information and context is important.
