var myApp = angular.module('stock-price-prediction',['angularMoment']);

myApp.controller('MarketController', ['$scope', '$http', 'moment', function($scope, $http, moment) {

    $scope.stocks = [];
    $scope.current_stock = {};
    
    readTextFile("/data/stocks.json", function(text){
        var data = JSON.parse(text);
        $scope.$apply(function () {
            $scope.stocks = data;
            var rand = Math.floor(Math.random() * data.length) + 1
            $scope.current_stock = $scope.stocks[rand];
        });
    });

    $http.get("/getstockdata/fdlj").then(function(response) {
        var data = JSON.parse(response.data).data;
        var date = new Date(data[0]["Date"]);

        var actualData = [];
        var predictedData = [];

        console.log(data[0]);

        for (var i = 0; i < data.length; i++) {
            actualData[i] = {};
            predictedData[i] = {};
            actualData[i].date = moment(data[i]["Date"]).format('YYYY-MM-DD');
            predictedData[i].date = moment(moment(data[i]["Date"]).format('YYYY-MM-DD'), "YYYY-MM-DD").add(1, 'years').format('YYYY-MM-DD');
            actualData[i].value = data[i]["EOD"];
            predictedData[i].value = data[i]["prediction"];
        }
        var chart1 = getChart("chartdiv1", actualData);
        chart1.addListener("rendered", zoomChart1);
        zoomChart1();
        
        function zoomChart1() {
            chart1.zoomToIndexes(chart1.dataProvider.length - 40, chart1.dataProvider.length - 1);
        }

        var chart2 = getChart("chartdiv2", predictedData);
        chart2.addListener("rendered", zoomChart2);
        zoomChart2();
        
        function zoomChart2() {
            chart2.zoomToIndexes(chart2.dataProvider.length - 40, chart2.dataProvider.length - 1);
        }
    });
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

    return AmCharts.makeChart(chart, {
        "type": "serial",
        "theme": "light",
        "marginRight": 40,
        "marginLeft": 40,
        "autoMarginOffset": 20,
        "mouseWheelZoomEnabled": true,
        "dataDateFormat": "YYYY-MM-DD",
        "valueAxes": [{
            "id": "v1",
            "axisAlpha": 0,
            "position": "left",
            "ignoreAxisWidth": true
        }],
        "balloon": {
            "borderThickness": 1,
            "shadowAlpha": 0
        },
        "graphs": [{
            "id": "g1",
            "balloon": {
                "drop": true,
                "adjustBorderColor": false,
                "color": "#ffffff"
            },
            "bullet": "round",
            "bulletBorderAlpha": 1,
            "bulletColor": "#FFFFFF",
            "bulletSize": 5,
            "hideBulletsCount": 50,
            "lineThickness": 2,
            "title": "red line",
            "useLineColorForBulletBorder": true,
            "valueField": "value",
            "balloonText": "<span style='font-size:18px;'>[[value]]</span>"
        }],
        "chartScrollbar": {
            "graph": "g1",
            "oppositeAxis": false,
            "offset": 30,
            "scrollbarHeight": 80,
            "backgroundAlpha": 0,
            "selectedBackgroundAlpha": 0.1,
            "selectedBackgroundColor": "#888888",
            "graphFillAlpha": 0,
            "graphLineAlpha": 0.5,
            "selectedGraphFillAlpha": 0,
            "selectedGraphLineAlpha": 1,
            "autoGridCount": true,
            "color": "#AAAAAA"
        },
        "chartCursor": {
            "pan": true,
            "valueLineEnabled": true,
            "valueLineBalloonEnabled": true,
            "cursorAlpha": 1,
            "cursorColor": "#258cbb",
            "limitToGraph": "g1",
            "valueLineAlpha": 0.2,
            "valueZoomable": true
        },
        "valueScrollbar": {
            "oppositeAxis": false,
            "offset": 50,
            "scrollbarHeight": 10
        },
        "categoryField": "date",
        "categoryAxis": {
            "parseDates": true,
            "dashLength": 1,
            "minorGridEnabled": true
        },
        "export": {
            "enabled": true
        },
        "dataProvider": data
    });
}