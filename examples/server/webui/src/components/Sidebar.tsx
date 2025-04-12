import { useEffect, useState } from 'react';
import { classNames } from '../utils/misc';
import { Conversation } from '../utils/types';
import StorageUtils from '../utils/storage';
import { useNavigate, useParams } from 'react-router';
import { useAppContext } from '../utils/app.context';

export default function Sidebar() {
  const params = useParams();
  const navigate = useNavigate();
  const { pendingMessages } = useAppContext();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currConv, setCurrConv] = useState<Conversation | null>(null);

  // Handler function for the clear all button
  const handleClearAll = async () => {
    const isAnyGenerating = Object.keys(pendingMessages).length > 0;
    if (isAnyGenerating) {
      alert(
        'Cannot clear conversations while message generation is in progress. Please wait or stop the generation.'
      );
      return; // Stop the function here
    }
    // Show confirmation dialog to the user
    const isConfirmed = window.confirm(
      'Are you sure you want to delete ALL conversations? This action cannot be undone.'
    );
    if (isConfirmed) {
      try {
        // Call the storage utility function to clear data
        await StorageUtils.clearAllConversations();
        // Navigate to the home/new conversation page after clearing
        // The onConversationChanged listener will handle updating the 'conversations' state automatically
        navigate('/');
      } catch (error) {
        console.error('Failed to clear conversations:', error);
        alert('Failed to clear conversations. See console for details.');
      }
    }
  };

  useEffect(() => {
    StorageUtils.getOneConversation(params.convId ?? '').then(setCurrConv);
  }, [params.convId]);

  useEffect(() => {
    const handleConversationChange = async () => {
      // Always refresh the full list
      setConversations(await StorageUtils.getAllConversations());

      // Check if the currently selected conversation still exists after a change (deletion/clear all)
      if (currConv?.id) {
        // Check if there *was* a selected conversation
        const stillExists = await StorageUtils.getOneConversation(currConv.id);
        if (!stillExists) {
          // If the current conv was deleted/cleared, update the local state for highlighting
          setCurrConv(null);
          // Navigation happens via handleClearAll or if user manually deletes and stays on the page
        }
      }
    };
    StorageUtils.onConversationChanged(handleConversationChange);
    handleConversationChange();
    return () => {
      StorageUtils.offConversationChanged(handleConversationChange);
    };
    // Dependency added to re-check existence if currConv changes while mounted
  }, [currConv]); // Changed dependency from [] to [currConv]

  return (
    <>
      <input
        id="toggle-drawer"
        type="checkbox"
        className="drawer-toggle"
        defaultChecked
      />

      <div className="drawer-side h-screen lg:h-screen z-50 lg:max-w-64">
        <label
          htmlFor="toggle-drawer"
          aria-label="close sidebar"
          className="drawer-overlay"
        ></label>
        <div className="flex flex-col bg-base-200 min-h-full max-w-[100%] py-4 px-4">
          <div className="flex flex-row items-center justify-between mb-4 mt-4">
            <h2 className="font-bold ml-4">Conversations</h2>

            {/* close sidebar button */}
            <label htmlFor="toggle-drawer" className="btn btn-ghost lg:hidden">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                fill="currentColor"
                className="bi bi-arrow-bar-left"
                viewBox="0 0 16 16"
              >
                <path
                  fillRule="evenodd"
                  d="M12.5 15a.5.5 0 0 1-.5-.5v-13a.5.5 0 0 1 1 0v13a.5.5 0 0 1-.5.5M10 8a.5.5 0 0 1-.5.5H3.707l2.147 2.146a.5.5 0 0 1-.708.708l-3-3a.5.5 0 0 1 0-.708l3-3a.5.5 0 1 1 .708.708L3.707 7.5H9.5a.5.5 0 0 1 .5.5"
                />
              </svg>
            </label>
          </div>

          {/* list of conversations */}
          <div
            className={classNames({
              'btn btn-ghost justify-start': true,
              'btn-active': !currConv,
            })}
            onClick={() => navigate('/')}
          >
            + New conversation
          </div>
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={classNames({
                // 'btn btn-ghost justify-start font-normal w-full overflow-hidden',
                'btn btn-ghost justify-start  font-normal': true,
                'btn-active': conv.id === currConv?.id,
                // Additional styles for active conversation
                'border-1 border-blue-400': conv.id === currConv?.id,
              })}
              onClick={() => navigate(`/chat/${conv.id}`)}
              dir="auto"
            >
              <span className="truncate">{conv.name}</span>
            </div>
          ))}
          <div className="text-center text-xs opacity-40 mt-auto mx-4 pb-2 ">
            Conversations are saved to browser's IndexedDB
          </div>
          {/* Clear All Button - Added */}
          {conversations.length > 0 && ( // Only show if there are conversations to clear
            <button
              className="btn btn-outline btn-error btn-sm w-full mb-3 pb-1"
              onClick={handleClearAll}
              title="Conversations are saved to browser's IndexedDB"
            >
              Clear All
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                fill="currentColor"
                className="bi bi-trash ml-2"
                viewBox="0 0 16 16"
              >
                <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0z" />
                <path d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4zM2.5 3h11V2h-11z" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </>
  );
}
