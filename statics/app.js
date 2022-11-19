class Chatbox { 
    constructor(){
        this.args = { //define different arguments for buttons, these selectors refer to class name from html
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false; //chatbox starting state
        this.messages = []; //array to store messages
    }

    display(){ //display messages
        const {openButton, chatBox, sendButton} = this.args; //extract arguments

        openButton.addEventListener('click', () => this.toggleState(chatBox)) //click on button to toggle state

        sendButton.addEventListener('click', () => this.onSendButton(chatBox)) //click on button to send message

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => { //if enter button entered
            if (key === "Enter"){
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox){
        this.state = !this.state; //switch state

        //show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active') //add or remove active class
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === ""){
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        fetch( $SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify( {message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value = ''
        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
        });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, ){
            if (item.name === "Same"){
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div'
            }
            else{
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div'
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();

