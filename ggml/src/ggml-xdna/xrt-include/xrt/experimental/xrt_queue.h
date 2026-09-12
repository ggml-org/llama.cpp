// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_QUEUE_H_
#define XRT_QUEUE_H_

#include "xrt/detail/config.h"

#ifdef __cplusplus
# include <algorithm>
# include <future>
# include <memory>
#endif

#ifdef __cplusplus

namespace xrt {

/**
 * class queue -- producer / consumer queue for tasks
 *
 * Used for sequencing operations in order of enqueuing.
 *
 * A queue has exactly one consumer which is a separate thread created
 * when the queue is constructed.
 *
 * When an opeation is enqueued on the queue an event is returned to
 * the caller.  This event can be enqueued in a different queue, which
 * will then wait for the former to complete the operaiton associated
 * with the event.
 */
class queue_impl;
class queue
{
  friend class queue_impl;

  // class task - type-erased callable operation
  //
  // A task wraps a caller's typed operation such that it
  // can be inserted into a queue.
  //
  // Tasks should be synchronous operations.
  class task
  {
    struct task_iholder
    {
      virtual ~task_iholder() {};
      virtual void execute() = 0;
    };

    // Wrap typed operation
    template <typename Callable>
    struct task_holder : public task_iholder
    {
      Callable m_held;

      explicit
      task_holder(Callable&& t)
        : m_held(std::move(t))
      {}

      void execute() override
      {
        m_held();
      }
    };

    std::unique_ptr<task_iholder> m_content;

  public:
    task() = default;
    task(task&& rhs) = default;

    // task() - task constructor for synchronous operation
    //
    // @c : callable object, a std::packaged_task
    template <typename Callable>
    task(Callable&& c)
      : m_content(new task_holder(std::forward<Callable>(c)))
    {}

    task&
    operator=(task&& rhs) = default;

    operator bool() const
    {
      return m_content != nullptr;
    }

    void
    execute()
    {
      m_content->execute();
    }
  };  // class queue::task

public:

  /**
   * class event - type-erased std::shared_future
   *
   * Wraps typed future return value returned when
   * enqueueing an operation.
   *
   * Returned futures implicitly convert to an event and as such
   * enqueue return values can be stored in an event container if
   * necessary.
   *
   * The event object is not needed where the typed future can be
   * used.
   */
  class event
  {
    struct event_iholder
    {
      virtual ~event_iholder() {};
      virtual void wait() const = 0;
    };

    // Wrap typed future
    template <typename ValueType>
    struct event_holder : public event_iholder
    {
      std::shared_future<ValueType> m_held;

      explicit
      event_holder(std::shared_future<ValueType>&& e)
        : m_held(std::move(e))
      {}

      void wait() const override
      {
        m_held.wait();
      }
    };

    std::shared_ptr<event_iholder> m_content;

  public:
    event() = default;
    event(event&& rhs) = default;
    event(const event& rhs) = default;

    // event() - event constructor for std::shared_future
    //
    // @e : shared_future to type erase
    template <typename ValueType>
    event(std::shared_future<ValueType> e)
      : m_content(std::make_shared<event_holder<ValueType>>(std::move(e)))
    {}

    event&
    operator=(event&& rhs) = default;

    operator bool() const
    {
      return m_content!=nullptr;
    }

    void
    wait() const
    {
      if (m_content)
        m_content->wait();
    }
  };

private:
  // Add task to queue
  XRT_API_EXPORT
  void
  add_task(task&& ev);

public:
  /**
   * queue() - Constructor for queue object
   *
   * The queue is constructed with one consumer thread.
   */
  XRT_API_EXPORT
  queue();

  /**
   * enqueue() - Enqueue a callable
   *
   * @param c
   *   Callable function, typically a lambda
   * @return
   *   Future result of the function (std::future)
   *
   * A callable is an argument-less lambda function.  The function is
   * executed asynchronously by the queue consumer (worker thread)
   * once all previous enqueued operations have completed.
   *
   * Upon completion the returned future becomes valid and will
   * contain the return value of executing the lambda.
   */
  template <typename Callable>
  auto
  enqueue(Callable&& c)
  {
    using return_type = decltype(c());
    std::packaged_task<return_type()> task{[cc = std::move(c)] { return cc(); }};
    std::shared_future f{task.get_future()};
    add_task(std::move(task));
    return f;
  }

  /**
   * enqueue() - Enqueue the future of an enqueued operation
   *
   * @param sf
   *   The future result to wait on (std::shared_future)
   * @return
   *   Future of the future (std::shared_future<void>)
   *
   * Subsequent enqueued task blocks until the enqueued future is
   * valid.
   *
   * This type of enqueued future is used for synchronization between
   * multiple queues.
   */
  template <typename ValueType>
  auto
  enqueue(std::shared_future<ValueType> sf)
  {
    return enqueue([evc = xrt::queue::event{std::move(sf)}] { evc.wait(); });
  }

  /**
   * enqueue() - Enqueue an event (type erased future)
   *
   * @oaram ev
   *   Event to enqueue
   * @param
   *   Future of event (std::shared_future<void>)
   *
   * Subsequent enqueued task blocks until the enqueued event is
   * valid.
   *
   * This type of enqueued event is used for synchronization between
   * multiple queues.
   */
  auto
  enqueue(xrt::queue::event ev)
  {
    return enqueue([evc = std::move(ev)] { evc.wait(); });
  }

public:
  queue_impl*
  get_impl() const
  {
    return m_impl.get();
  }

private:
  std::shared_ptr<queue_impl> m_impl;
};

} // namespace xrt

#else
# error xrt_enqueue is only implemented for C++
#endif // __cplusplus

#endif
