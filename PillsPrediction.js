//i used npm install @tensorflow/tfjs-node to install the tensorflowjs library

const tf = require('@tensorflow/tfjs-node');
const { sequelize } = require("./models");
sequelize .sync();
const {User, Pill, Date} = require("./models");

  async function save (model) {
    await model.save(`file://./pillpredictionmodel`);
  }

//function to load a model saved on local storage with the save function
//could be modified to be able to load a model saved on an online server
async function load (model){
      const models = await tf.io.listModels();
      const modelInfo = models['file://./pillpredictionmodel'];
      return model = await tf.loadLayersModel('file://./pillpredictionmodel/model.json');

  }
// we initialize some variables needed for most of the following functions
  let points
  let model = null;
  let normalisedFeature, normalisedLabel;
  let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;

async function modelTraining(points){
  //Extract Features (inputs)
    const featureValues = points.map(p => [p.a,p.b,p.c,p.d,p.e,p.f,p.g,p.h,p.i,p.j,p.k,p.l,p.m,p.n]);
    const featureTensor = tf.tensor2d(featureValues);


    //Extract Labels (outputs)
    const labelValues = points.map(p => getClassIndex(p.class));
    const labelTensor = tf.tidy(() => tf.oneHot(tf.tensor1d(labelValues, 'int32'), 12));

    [trainingFeature, testingFeature] = tf.split(featureTensor, 2);
    [trainingLabel, testingLabel] = tf.split(labelTensor, 2);
     model = createLinearModel();

  const result = await trainModel(model, trainingFeature, trainingLabel);


}


  async function predict(predictionInput) {
      const inputTensor = tf.tensor2d([predictionInput]);
      const outputValue = getClassName(tf.argMax(model.predict(inputTensor),axis=1));
      return outputValue;

  }

  function createLinearModel(){
   const model = tf.sequential();
   model.add(tf.layers.dense({
     units: 28,
     useBias: true,
     activation: 'sigmoid',
     inputDim: 14,
   }));
   model.add(tf.layers.dense({
     units: 56,
     useBias: true,
     activation: 'relu',
   }));
   model.add(tf.layers.dense({
     units: 28,
     useBias: true,
     activation: 'sigmoid',
   }));
   model.add(tf.layers.dense({
     units: 12,
     useBias: true,
     activation: 'softmax',
   }));
   const optimizer = tf.train.adam();
   model.compile({
     loss: 'categoricalCrossentropy',
     optimizer,
   })
   return model;
 }

  async function trainModel (model, trainingFeature, trainingLabel){

    return model.fit(trainingFeature,trainingLabel, {
      batchSize: 32,
      epochs: 200,

    });
  }


function getClassIndex(className){
  if(className==1 || className==="1"){return 0;}
  else if(className==2 || className==="2"){return 1;}
  else if(className==3 || className==="3"){return 2;}
  else if(className==4 || className==="4"){return 3;}
  else if(className==5 || className==="5"){return 4;}
  else if(className==6 || className==="6"){return 5;}
  else if(className==7 || className==="7"){return 6;}
  else if(className==8 || className==="8"){return 7;}
  else if(className==9 || className==="9"){return 8;}
  else if(className==10 || className==="10"){return 9;}
  else if(className==11 || className==="11"){return 10;}
  else if(className==12 || className==="12"){return 11;}
}

function getClassName(classIndex){
  value = classIndex.dataSync();
  output = +value + 1;
  return (output);
}

//this function access the pillprediction dataset (here in csv form as a local file)
//in order to train a model capable of predicting which pill to use according to a users input
  async function modelTrainingOnDataset(){
    //import dataset from CSV, can be replace by accessing the same dataset saved on the mySQL database
    //The most convenient solution would be to train it once and save the model on the server
    //This way we wont have to retrain it at all. We could simply load it from an online server, pretrained
    //to just use it with the users data to make a prediction
    // To summarize this function can be used once in the project to create the model,save it and put it on an online server
    const dataset = tf.data.csv('file://./ds_pills.csv');

    //We extract all values from the dataset (in csv format)
    const pointsDataset = dataset.map(record => ({
      a: record.backache,
      b: record.brash,
      c: record.throes,
      d: record.menstrualirregularity,
      e: record.swell,
      f: record.paininlowerabdomen,
      g: record.slightfever,
      h: record.headache,
      i: record.stomachache,
      j: record.convulsion,
      k: record.mentalproblem,
      l: record.diarrhoea,
      m: record.alcohol,
      n: record.gastrointestinaldisturbance,
      class: record.PillNumber,
    }));
    points = await pointsDataset.toArray();
    if(points.length % 2 !== 0) {points.pop();}
    tf.util.shuffle(points);

    await modelTraining(points);
    await model.save(`file://./pillpredictionmodel`);

}

//this function access the symptoms dataset of a single user (here in csv form as a local file)
//in order to use the pre trained model and make a predicting on the pill to take based on the symptoms
  async function modelPredictingOnUserDataset(userId){
    //import dataset of a user from CSV, can be replace by accessing the same dataset saved on the mySQL database
    // Here as we need to add the access to the mySQL database, we will still use the pillprediction dataset saved as csv file
    // We will access it and use one of the rows as an input for the predictionInput
    //But all of the following can be replace by accessing the data of a user via mySQL


    const symptoms = await Date.findAll({ where:{ userId: userId,}});
    const nbDates = await Date.count({where:{ userId: userId,}});

    const featuresarray = new Array(14);
    for(var i=0; i<featuresarray.length;i++){
      featuresarray[i]=0;
    }
    if(symptoms[nbDates-1]['dateCondition1'])
    {
      featuresarray[symptoms[nbDates-1]['dateCondition1']]=1;
    }
    if(symptoms[nbDates-1]['dateCondition2'])
    {
      featuresarray[symptoms[nbDates-1]['dateCondition1']]=1;
    }
    if(symptoms[nbDates-1]['dateCondition3'])
    {
      featuresarray[symptoms[nbDates-1]['dateCondition1']]=1;
    }

    model = await load(model);
    var pill = await predict(featuresarray);
    await Date.update({ pillReco: pill}, {
  where: {
    id: nbDates,
    userId: userId,
  }
});


}
//modelTrainingOnDataset();
modelPredictingOnUserDataset(1);

module.exports = {modelPredictingOnUserDataset};
//use modelPredictingOnUserDataset(userID) to make a prediction;
