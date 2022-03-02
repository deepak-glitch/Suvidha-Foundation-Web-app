function sendEmail(){
  Email.send({
    Host : "smtp.gmail.com",
    Username : "username",
    Password : "password",
    To : 'Info@Suvidhafoundationedutech.Org',
    From : document.getElementsByClassName("input2").value,
    Subject : "Enquiry",
    Body : "Name: " + document.getElementsByClassName("input1").value + "\n" + "Email: " + document.getElementsByClassName("input2").value + "\n" + "Phone: " + document.getElementsByClassName("input3").value +  "\n" +"Message: " + document.getElementsByClassName("input3").value
  }).then(
  message => alert("Message Sent Successfully")
  );
}