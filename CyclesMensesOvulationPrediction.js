//i used npm install @tensorflow/tfjs-node to install the tensorflowjs library

const tf = require('@tensorflow/tfjs-node');
const { sequelize } = require("./models");
sequelize .sync();
const {User, Predict, Cycle} = require("./models");


//this storageID can be used when saving the model or loading it, however we should need to use it as we will train a new model everytime we update data of the user to predict values with more precision with the new data
const storageID = "cycles menses ovulation prediction";
  //function to save a model, we shouldnt have to use it here
  async function save (model) {
    await model.save(`file://./model`);
  }

  //function to load a model saved on local storage with the save function, we shouldnt have to use it here
  async function load (model){
        const models = await tf.io.listModels();
        const modelInfo = models['file://./ovulationdaymodel'];
        return model = await tf.loadLayersModel('file://./ovulationdaymodel/model.json');

    }

// we initialize some variables needed for most of the following functions
  let points
  let model = null;
  let timePortion=4;
  let min;
  let max;

  /*what this function does is getting the data from the user which will be :
  - an array with all the previous lengths of cycles of a user in order to predict the length of the next cycle
  - an array with all the previous lengths of menses of a user in order to predict the length of the next menses
  - an array with all the preivous ovulationDays of a user in order to predict the length of the next menses
  */
async function modelTrainingAndPredictingOnDataset(data, timePortion) {
       let trainX = [], trainY = [], size = data.length;
       let features = [];
        for (let i = 0; i < size; i++) {
            features.push(data[i]);
        }

// Then the data is transformed as follows :

// we scale the data in order to normalize it before using it with our model.


        var scaledData = minMaxScaler(features, getMin(features), getMax(features));
        let scaledFeatures = scaledData.data;
        min = scaledData.min;
        max = scaledData.max;

// - Then we prepare our training features and training labels.
// As we are using a 1D CNN as our model, at each timestep the input is the following :
// n number of previous cycleLength as features, and the next cycleLength as the label.
// Therefore we go through the whole dataset with a double for loop, in order to get n = timePortion values as features and the next value as the label, starting at i= timePortion and j= i- timePortion.
// which means that the values of scaledFeatures[i] will be the labels and the values of scaledFeatures[j] will be the features.

        for (let i = timePortion; i < size; i++) {
              for (let j = (i - timePortion); j < i; j++) {
                    trainX.push(scaledFeatures[j]);
              }
              trainY.push(scaledFeatures[i]);
          }

//Then we create tensors based on the features (trainX) and the labels (trainY).
//Our 1D CNN model is expecting a 3d tensor as an input, which is why we redimension it with the tf.reshape method as it is only a flat array right now.
// The 3d tensor as the following shape : [length of the 3d tensor : number of datapoints = size-timePortion, number of features : window of previous cycles = timePortion, 1]

 const tensorTrainX = await tf.tensor1d(trainX).reshape([size-timePortion, timePortion, 1]);
 const tensorTrainY = await tf.tensor1d(trainY);

// here we create the model using the build1DCNN() function that you can find below.
model = build1DCNN(timePortion);

// here with the method tf.fit, we train the model on a certain number of epochs using our tensors of features and labels.
await model.fit(tensorTrainX,tensorTrainY, {epochs: 500,});
return predictNextMonthValue(generateNextDayPrediction(features, timePortion),model,min,max,timePortion);

}


//The following function is used to create the AI model
// As it is widely used for time prediction problems, we decided to use a 1D CNN model. It is composed of an input layer, a convolutional layer, a flatten layer and a dense layer as our output layer
function build1DCNN(timePortion){
  // Linear (sequential) stack of layers
var kernel=3;
if(timePortion=3){kernel=2;}


const model = tf.sequential();

// Define input layer
model.add(tf.layers.inputLayer({
    inputShape: [timePortion,1],
}));

// Add the first convolutional layer
model.add(tf.layers.conv1d({
    kernelSize: kernel,
    filters: 120,
    strides: 1,
    use_bias: true,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
}));


// Add Flatten layer, reshape input to (number of samples, number of features)
model.add(tf.layers.flatten({
}));

// Add Dense layer,
model.add(tf.layers.dense({
    units: 1,
    kernelInitializer: 'VarianceScaling',
    activation: 'linear'
}));
// we then compile the model using an adam optimizer and mean squared error as our loss function.
model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
return model;
}

//this function creates the necessary features to predict the next cycleLength, mensesLength or ovulationDay
//those features being the n = timePortion points of the dataset, which are the n = timePortion last values of cycleLength, mensesLength or ovulationDay
const generateNextDayPrediction = function (data, timePortion) {
    let size = data.length;
    let features = [];

    for (let i = (size - timePortion); i < size; i++) {
        features.push(data[i]);
    }

    return features;
}

async function predictOvulation(predictionInput) {
    const inputTensor = tf.tensor2d([predictionInput],[1,1]);
    const outputValue = tf.argMax(model.predict(inputTensor), axis=1);
    const output = outputValue.dataSync();
    return output;

}

//the predictNextMonthValue function is used to predict the value of next cycleLength, mensesLength or ovulationDay
async function predictNextMonthValue(predictionInput, model, min, max,timePortion){
  //we first scale the predictionInput array using minMaxScaler
  let nextMonth = minMaxScaler(predictionInput, min, max);
  //then we create a tensor based on nextMonth array and reshape it as a 3d tensor as needed by the model
  let tensorNextMonth = tf.tensor1d(nextMonth.data).reshape([1,timePortion,1]);
  //we used tf.predict method to predict the output value using the tensorNextMonth
  let predictedValue = model.predict(tensorNextMonth);
  // we rescale the output value of the model using minMaxInverseScaler
  var inversePredictedValue = minMaxInverseScaler(predictedValue.dataSync()[0], min, max);
  // the value calculated by the model is a float value, but as we want an int value (the output being a length of days of a day), we round it.
  var output =  Math.round(inversePredictedValue);
  return output;
}

//this function is using to scale/normalize our data based on the min and max value of the dataset, this is called min-max normalization
const minMaxScaler = function (data, min, max) {
    let scaledData = data.map(function (value) {
        return (value - min) / (max - min);
    });
    return {
        data: scaledData,
        min: min,
        max: max
    }
}

// this function does the opposite and reverts the min-max normalization to get the real values
const minMaxInverseScaler = function (value, min, max) {
        return value * (max - min) + min;
}

//Get min value from array
const getMin = function (data) {
    return Math.min(...data);
}

//Get max value from array
const getMax = function (data) {
    return Math.max(...data);
}
//this function is the one using all previous functions and accessing the dataset being a csv local file for the moment and needing to be replace by accessing the mySQL database
async function predictAllValues(userId){
 //we access the cycles of a user using his userID
  const cyclesquery = await Cycle.findAll({ where:{ userId: userId,}});
  const cycles = cyclesquery.map(c => c.get({plain:true}));
  var nbOfCycles = cycles.length-1;

  // we initialize 3 arrays for each features that we will use to train and predict the 3 next values
  var cyclelengtharray = Array.from(Array(nbOfCycles), () => new Array(1));
  var menseslengtharray = Array.from(Array(nbOfCycles), () => new Array(1));
  if(nbOfCycles=4){timePortion=3;}
  if(nbOfCycles>=4)
  {
    for(i = 1; i<cycles.length; i++){
      //we create a date object using the string value of the database
      var bleedStart = new Date(cycles[i-1]['bleedStart']);
      var bleedStart2 = new Date(cycles[i]['bleedStart']);
      //then we calculate the legnthOfCycle
      var lengthOfCycle = Math.round((bleedStart2.getTime() - bleedStart.getTime())/ (1000 * 3600 * 24));
      // we then fill the arrays with the values calculated
      cyclelengtharray[i-1][0]=lengthOfCycle;
    }

    //we fill the arrays previously created with the values found in the database
    for(i = 0; i<cycles.length-1; i++){
      //we create a date object using the string value of the database
      var bleedStart = new Date(cycles[i]['bleedStart']);
      var bleedEnd = new Date(cycles[i]['bleedEnd']);
      //then we calculate the lengthOfPeriods
      var lengthOfPeriods = Math.round((bleedEnd.getTime() - bleedStart.getTime())/ (1000 * 3600 * 24));
      // we then fill the array with the values calculated
      menseslengtharray[i][0]=lengthOfPeriods;
    }


    console.log(menseslengtharray);

    // we use it to predict the value of the next cycleLength
    var nextCycleLength = await modelTrainingAndPredictingOnDataset(cyclelengtharray, timePortion);
    cyclelengtharray.push([nextCycleLength]);
    var nextCycleLength2 = await modelTrainingAndPredictingOnDataset(cyclelengtharray, timePortion);

    // we use it to predict the value of the next cycleLength
    var nextMensesLength = await modelTrainingAndPredictingOnDataset(menseslengtharray, timePortion);
    menseslengtharray.push([nextMensesLength]);
    var nextMensesLength2 = await modelTrainingAndPredictingOnDataset(menseslengtharray, timePortion);

    // here the last task needed is to send those 3 values to the mySQL database

  //we count the previous predictions of this user saved in the database so we dont try to
  // overwrite existing values
    const predicts = await Predict.count({where:{ userId: userId,}});


    // here we initialize the new predicted dates, they are all copies of the end date of the preivous cycle
    // to which we add the number of days corresponding to our predictions made by the models
    var predictedBleedStart = new Date(Number(bleedStart2));
    var predictedBleedEnd = new Date(Number(bleedStart2));
    var predictedEggStart = new Date(Number(bleedStart2));
    var predictedEggEnd = new Date(Number(bleedStart2));

    model = await load(model);
    const inputTensor = tf.tensor2d([nextCycleLength2],[1,1]);
    const outputValue = model.predict(inputTensor);
    const predictedOvulationDay = outputValue.dataSync()[0];

    predictedBleedStart.setDate(predictedBleedStart.getDate()+nextCycleLength);
    predictedBleedEnd.setDate(predictedBleedEnd.getDate()+(nextCycleLength+nextMensesLength2));
    predictedEggStart.setDate(predictedEggStart.getDate()+(nextCycleLength+predictedOvulationDay-3));
    predictedEggEnd.setDate(predictedEggEnd.getDate()+(nextCycleLength+predictedOvulationDay+3));

  //we create a new predict in the database using the predicted dates
    await Predict.create({
      id: predicts+1,
      predictBleedStart: predictedBleedStart.toISOString().substring(0,10),
      predictBleedEnd: predictedBleedEnd.toISOString().substring(0,10),
      predictEggStart: predictedEggStart.toISOString().substring(0,10),
      predictEggEnd: predictedEggEnd.toISOString().substring(0,10),
      userId: userId,
    });
  }



}

module.exports = { predictAllValues};
//predictAllValues(1);

//user predictAllValues(userId) to make a prediction
