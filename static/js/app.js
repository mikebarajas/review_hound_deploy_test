
$(function () {
    $('button').click(function() {
        var userReview = $('#userInput').val();
        $.ajax({
            url: '/prediction',
            data: $('form').serialize(),
            type: 'POST',
            success: function(response) {
                $('#result').text('Review Hound found the review to be:');
                $('#result2').text((response*100).toFixed(2) + '% Positive');
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});            
