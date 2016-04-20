/**
 * Created by rocky on 4/18/16.
 */

function select(btn) {
    $("#proposal").css('background-color', 'rgba(255, 255, 255, 0.08)');
    $("#checkpoint").css('background-color', 'rgba(255, 255, 255, 0.08)');
    btn.css('background-color', 'rgba(255, 255, 255, 0.3)')
}

$(document).ready(function () {
    $(".main-content").load("checkpoint.html");
    select($("#checkpoint"));

    $("#proposal").click(function () {
        $(".main-content").load("proposal.html");
        select($("#proposal"));
    });

    $("#checkpoint").click(function () {
        $(".main-content").load("checkpoint.html");
        select($("#checkpoint"));
    })
});
