var hnnet = ee.FeatureCollection("users/hlahiouel/nnet");

//// function for using neural network
var nnet_class=function(nnet, img){
    // function for parsing neural network layers
    var parse_layer = function(feature) {
      feature = ee.Feature(feature);
      var prev_layer_size = feature.getNumber("prev_layer_size");
      var num_nodes = feature.getNumber("num_nodes");
      var node_size = prev_layer_size.subtract(1);
      var activation = feature.getString("activation");
      
      var node_collection = ee.ImageCollection(ee.List.sequence(1, num_nodes).map(function(node) {
        node = ee.Number(node).toInt();
        return ee.ImageCollection(ee.List.sequence(node, node.add(node_size.multiply(num_nodes)), num_nodes)
                        .map(function(index) {
                          index = ee.Number(index).toInt();
                          return ee.Image(feature.getNumber(index));
                        })).toBands().set({"bias": 
                          feature.getNumber(node.add(prev_layer_size.multiply(num_nodes)))
                        });
      }));
      
      return ee.List([node_collection, activation]);
    };
    // different activate functions
    var linear = function(x) {
      return ee.Image(x);
    };
    var elu = function(x) {
      x = ee.Image(x);
      return ee.ImageCollection([x.mask(x.gte(0)), x.mask(x.lt(0)).exp().subtract(1)]).mosaic();
    };
    var softplus = function(x) {
      x = ee.Image(x);
      return x.exp().add(1).log();
    };
    var softsign = function(x) {
      x = ee.Image(x);
      return x.divide(x.abs().add(1));
    };
    var relu = function(x) {
      x = ee.Image(x);
      return x.max(0.0);
    };
    var tanh = function(x) {
      x = ee.Image(x);
      return x.multiply(2).exp().subtract(1).divide(x.multiply(2).exp().add(1));
    };
    var sigmoid = function(x) {
      return x.exp().pow(-1).add(1).pow(-1);
    };
    //function for applying neural network in image for prediction
    var apply_nnet = function(layer, input) {
      layer = ee.List(layer);
      input = ee.Image(input);
      
      var layer_nodes = ee.ImageCollection(layer.get(0));
      var activation = layer.getString(1);
     
      var node_outputs =layer_nodes.map(function(node){ 
        node = ee.Image(node);
        var bias=ee.Algorithms.If(node.getNumber('bias'),node.getNumber('bias'),0.0)
        var result =  input.multiply(node).reduce(ee.Reducer.sum()).add(ee.Number(bias));
        return ee.Algorithms.If(activation.compareTo("linear"), softsign(result), result);
     }).toBands();
      
      return node_outputs;
    };
    
    //filter network coeffients 
    nnet = nnet.filterBounds(ee.Geometry.Point([0,0]));
    var layer_list = nnet.sort("layer_num").toList(nnet.size());
    var neural_net = layer_list.map(parse_layer);
  
    // apply neural network in image input
    var prediction_data = ee.Image(neural_net.iterate(apply_nnet, img)).rename("NNET");
    return prediction_data;
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////data preparation/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
var inputImage = ee.Image('COPERNICUS/S2_SR/20200811T164849_20200811T165525_T16UEA');
print (inputImage)
Map.centerObject(inputImage)
var proj20m =inputImage.select('B5').projection();
var proj10m =inputImage.select('B3').projection();
//// resample bands to 20m
var resamp1=ee.ImageCollection(inputImage.select('B[3-4]','B8')).map(function(image) { return image.reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .reproject({
      crs: proj20m
    })
    });

//print (resamp1)
var geometry=inputImage.select('B7').geometry()
/// combine 20m bands 
var inputImage2=resamp1.select('B[3-4]','B8').combine(inputImage.select('B[5-7]','B8A','B11', 'B12'))
//print (inputImage2)
/// get the image from image collection
inputImage=inputImage2.first()
/// calculate NDVI
var input_nvdi = (inputImage.select('B8').subtract(inputImage.select('B4'))).divide(inputImage.select('B8').add(inputImage.select('B4')).add(ee.Image.constant(0.01))).rename('NDVI')
/// add angle parameters
var img = inputImage.addBands(inputImage.metadata('MEAN_INCIDENCE_ZENITH_ANGLE_B8A').multiply(3.1415).divide(180).cos().multiply(10000).toUint16().setDefaultProjection(proj10m,null ,20).rename(['cosVZA']))
                    .addBands(inputImage.metadata('MEAN_SOLAR_ZENITH_ANGLE').multiply(3.1415).divide(180).cos().multiply(10000).toUint16().setDefaultProjection(proj10m,null ,20).rename(['cosSZA']))
                    .addBands(inputImage.metadata('MEAN_SOLAR_AZIMUTH_ANGLE').subtract(inputImage.metadata('MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A')).multiply(3.1415).divide(180).cos().multiply(10000).toInt16().setDefaultProjection(proj10m,null,20).rename(['cosRAA']))
                    .addBands(input_nvdi);

//print (img, 'img')
// resample angle parameter to 20m
var resamp2=ee.ImageCollection(img.select('cosVZA','cosSZA')).map(function(image) { return image.reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .reproject({
      crs: proj20m
    })
    });
    
//print (resamp2, 'resamp2')
/// select image bands and divide by 10000
var inputImage_selected = img.select('B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12', 'cosVZA', 'cosSZA', 'cosRAA', 'NDVI').divide(10000);

// Hardcode the mean and std for each input
var mean_inputs = ee.Image.constant([0.08782603761118264, 0.04438705689972076, 0.12059584827039747, 0.3827791870304996, 0.5085891428901771, 0.5281779262809694, 0.22814188690644405, 0.1084917037763612, 0.98307462649649, 0.7184247776912331, 0.019621872151453095, 0.15850844179262216]);
var std_inputs = ee.Image.constant([0.03745300784427652, 0.025190186604935685, 0.04757612965923997, 0.1155960480541525, 0.15462569909256219, 0.15759619292988825, 0.07959300198646108, 0.058194572071145007, 0.012550949423169259, 0.15160681873160317, 0.7631794471694554, 0.05361330338873914]);
// Standardize the data 
var scaledImage = inputImage_selected.subtract(mean_inputs).divide(std_inputs);
scaledImage=ee.ImageCollection(scaledImage).map(function(image){return image.clip(geometry)})
// select image bands for input
var selected_features = ee.List(['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12','cosVZA', 'cosSZA', 'cosRAA', 'NDVI']);
var img_inputs = scaledImage.select(selected_features).first();
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////apply generic function///////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// get the neural network
var nnet=hnnet;
// prediction using generic function defined
var prediction=nnet_class(nnet, img_inputs);
print(prediction);
Map.addLayer(prediction, {min: 0, max: 10}, 'prediction');

