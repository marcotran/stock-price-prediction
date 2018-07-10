var myApp = angular.module('stock-price-prediction',['angularMoment']);

myApp.controller('MarketController', ['$scope', '$http', 'moment', function($scope, $http, moment) {

    $scope.stocks = [];
    $scope.methods = [];
    $scope.current_stock = {
        "name": "Apple Inc.",
        "ticker": "AAPL"
    };
    $scope.current_method = 1;

    var chart_config = {};
    
    readTextFile("/data/methods.json", function(text){
        var data = JSON.parse(text);
        $scope.$apply(function () {
            $scope.methods = data;
            readTextFile("/data/stocks.json", function(text){
                var data = JSON.parse(text);
                $scope.$apply(function () {
                    $scope.stocks = data;
                    var rand = Math.floor(Math.random() * data.length) + 1
                    $scope.current_stock = $scope.stocks[rand];
                    readTextFile("/data/chart_config.json", function(text){
                        var data = JSON.parse(text);
                        chart_config = data;
                        setStockData();
                    });
                });
            });
        });
        var method1 = document.querySelector('#exampleRadios1');
        method1.setAttribute('checked', true);
    });

    $scope.changeStock = function(stock) {
        if ($scope.current_stock != stock) {
            console.log(`changing stock to ${stock.name}`);
            $scope.current_stock = stock;
            setStockData();
        }
    }

    $scope.changeMethod = function(method) {
        if ($scope.current_method != method) {
            console.log(`changing method to ${method}`);
            $scope.current_method = method;
            setStockData();
        }
    }

    function setStockData() {
        $http.get(`/getstockdata/?stock=${$scope.current_stock.ticker}&method=${$scope.current_method}`).then(function(response) {
            var data = JSON.parse(response.data).data;

            var actualData = [];
            var predictedData = [];

            for (var i = 0; i < data.length; i++) {
                actualData[i] = {};
                predictedData[i] = {};
                actualData[i].date = moment(data[i]["Date"]).format('YYYY-MM-DD');
                predictedData[i].date = moment(moment(data[i]["Date"]).format('YYYY-MM-DD'), "YYYY-MM-DD").add(1, 'years').format('YYYY-MM-DD');
                actualData[i].value = data[i]["EOD"];
                predictedData[i].value = data[i]["prediction"];
            }
            var chart_config_1 = JSON.parse(JSON.stringify(chart_config));
            chart_config_1.dataProvider = actualData;
            var chart1 = getChart("chartdiv1", chart_config_1);
            chart1.addListener("rendered", zoomChart1);
            zoomChart1();
            
            function zoomChart1() {
                chart1.zoomToIndexes(chart1.dataProvider.length - 40, chart1.dataProvider.length - 1);
            }

            var chart_config_2 = JSON.parse(JSON.stringify(chart_config));
            chart_config_2.dataProvider = predictedData;
            var chart2 = getChart("chartdiv2", chart_config_2);
            chart2.addListener("rendered", zoomChart2);
            zoomChart2();
            
            function zoomChart2() {
                chart2.zoomToIndexes(chart2.dataProvider.length - 40, chart2.dataProvider.length - 1);
            }
        });
    }
}]);

function readTextFile(file, callback) {
    var rawFile = new XMLHttpRequest();
    rawFile.overrideMimeType("application/json");
    rawFile.open("GET", file, true);
    rawFile.onreadystatechange = function() {
        if (rawFile.readyState === 4 && rawFile.status == "200") {
            callback(rawFile.responseText);
        }
    }
    rawFile.send(null);
}

function getChart(chart, data) {
    return AmCharts.makeChart(chart, data);
}