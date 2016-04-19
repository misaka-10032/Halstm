/**
 * Created by rocky on 4/18/16.
 */

function select(btn) {
    $("#proposal").css('background-color', 'rgba(255, 255, 255, 0.08)');
    $("#midway").css('background-color', 'rgba(255, 255, 255, 0.08)');
    btn.css('background-color', 'rgba(255, 255, 255, 0.3)')
}

$(document).ready(function () {
    $(".main-content").load("proposal.html");
    select($("#proposal"));

    $("#proposal").click(function () {
        $(".main-content").load("proposal.html");
        select($("#proposal"));
    });
    
    $("#midway").click(function () {
        $(".main-content").load("midway.html");
        select($("#midway"));
    })
});
