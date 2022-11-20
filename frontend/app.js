// Chatbox object handles event features for the chat box, presenting 
// conversation with chat bot preferrable for UI

class Chatbox {
    constructor() {
        // arguments passed into the constructor
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }
        

        this.state = false; // for box
        this.messages = []; // collects all the user messages
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    // simple toggle button effect
    toggleState(chatbox) {
        this.state = !this.state;
        // show or hides the box
        if (this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    // effect for send
    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text= textField.value
        if (text === "") {
            return;
        }

        let msg = { name: "User", message: text }
        this.messages.push(msg);
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text }),
            // to activate Flask API that supports without external host from a library
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(r => {
                let msg_bot = { name: "Suzanne", message: r.answer };
                this.messages.push(msg_bot);
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
        this.messages.slice().reverse().forEach(function (item) {
            if (item.name === "Suzanne") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}


//create the instance of chatbox object
const chatbox = new Chatbox();
chatbox.display();