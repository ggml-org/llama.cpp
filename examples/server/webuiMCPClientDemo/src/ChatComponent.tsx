// ChatComponent.tsx
import React from 'react';

interface ChatComponentProps {
  messages: { role: string, content: string }[];
  input: string;
  setInput: (input: string) => void;
  onSendMessage: () => void;
}

const ChatComponent: React.FC<ChatComponentProps> = ({ messages, input, setInput, onSendMessage }) => {
  return (
    <div>
      <div style={{ height: '300px', overflowY: 'scroll', border: '1px solid #ccc', padding: '10px', marginBottom: '10px' }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ marginBottom: '5px', textAlign: msg.role === 'user' ? 'right' : 'left' }}>
            <strong>{msg.role === 'user' ? 'You:' : 'Assistant:'}</strong> {msg.content}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && onSendMessage()}
        style={{ width: '70%', padding: '5px', marginRight: '5px' }}
      />
      <button onClick={onSendMessage} style={{ padding: '5px 10px' }}>Send</button>
    </div>
  );
};

export default ChatComponent;
