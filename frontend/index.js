
(function () {
    'use strict';

    var init = function () {                
        var slider3 = new rSlider({
            target: '#range-slider',
            values: {min: 0, max: 100},
            step: 1,
            range: true,
            set: [10, 20],
            scale: false,
            labels: false,
            onChange: function (vals) {
                console.log(vals);
            }
        });
    };
    
    window.onload = init;
})();


