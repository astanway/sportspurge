var count=0,refresh=function(){var e=$(".js-stream-item");if(count!=e.length){count=e.length;for(var t={},n=0;n<e.length;n++){var r=$(e[n]).attr("id");if(r.indexOf("tweet")==-1){continue}t[r]=$("#"+r).find(".js-tweet-text")[0].innerText}$.ajax({type:"POST",dataType:"json",url:"https://sportspurge.com/tweet",data:JSON.stringify(t),success:function(e){for(var t=0;t<e.length;t++)$("#"+e[t]).remove()}});count=e.length}};$(document).ready(function(){window.setInterval(refresh,10)})